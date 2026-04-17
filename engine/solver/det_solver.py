"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import time
import json
import datetime
import math

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):
    def _safe_float(self, value, default=float("nan")):
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except Exception:
            return float(default)

    def _point_checkpoint_cfg(self):
        cfg = getattr(self.cfg, "yaml_cfg", {}).get("point_checkpointing", {})
        return cfg if isinstance(cfg, dict) else {}

    def _point_checkpoint_enabled(self):
        return bool(self._point_checkpoint_cfg().get("enabled", False))

    def _phase2_cfg(self):
        schedule = getattr(self.cfg, "yaml_cfg", {}).get("point_training_schedule", {})
        if not isinstance(schedule, dict):
            return {}
        phase2 = schedule.get("phase2", {})
        return phase2 if isinstance(phase2, dict) else {}

    def _checkpoint_metrics_path(self):
        return self.output_dir / "point_checkpoint_metrics.json"

    def _point_pair_count(self, metrics):
        if not isinstance(metrics, dict):
            return 0
        return max(int(self._safe_float(metrics.get("valid_point_pair_count", 0.0), 0.0)), 0)

    def _has_valid_point_l2_candidate(self, metrics):
        if not isinstance(metrics, dict):
            return False
        point_l2 = self._safe_float(metrics.get("valid_point_mean_L2_px"))
        return self._point_pair_count(metrics) > 0 and math.isfinite(point_l2)

    def _load_point_checkpoint_state(self):
        default = {
            "config": self._point_checkpoint_cfg(),
            "best_grape_ap": {"epoch": -1, "score": float("-inf"), "metrics": {}},
            "best_has_picking_f1": {"epoch": -1, "score": float("-inf"), "metrics": {}},
            "best_point_l2": {"epoch": -1, "score": float("inf"), "metrics": {}},
            "best_composite": {"epoch": -1, "score": float("-inf"), "metrics": {}},
        }
        path = self._checkpoint_metrics_path()
        if not path.exists():
            return default
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
        if not isinstance(loaded, dict):
            return default
        sanitized = False
        for key, value in default.items():
            loaded.setdefault(key, value)
        best_point_l2_state = loaded.get("best_point_l2", {})
        if not self._has_valid_point_l2_candidate(best_point_l2_state.get("metrics", {})):
            loaded["best_point_l2"] = {
                "epoch": -1,
                "score": float("inf"),
                "metrics": {},
            }
            sanitized = True
        if sanitized and dist_utils.is_main_process():
            payload = dict(loaded)
            payload["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            payload["config"] = self._point_checkpoint_cfg()
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return loaded

    def _save_point_checkpoint_state(self):
        if not self._point_checkpoint_enabled():
            return
        payload = dict(self.point_checkpoint_state)
        payload["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        payload["config"] = self._point_checkpoint_cfg()
        self._checkpoint_metrics_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _extract_point_eval_metrics(self, test_stats):
        cfg = self._point_checkpoint_cfg()
        grape_class_name = str(cfg.get("grape_class_name", "grape"))
        alpha = self._safe_float(cfg.get("alpha", 0.35), 0.35)
        beta = self._safe_float(cfg.get("beta", 0.2), 0.2)
        point_error_norm_px = max(self._safe_float(cfg.get("point_error_norm_px", 40.0), 40.0), 1e-6)

        per_class = test_stats.get("coco_eval_bbox_per_class", {})
        grape_metrics = per_class.get(grape_class_name, {}) if isinstance(per_class, dict) else {}
        bbox_metrics = test_stats.get("coco_eval_bbox", [])
        grape_ap = self._safe_float(
            grape_metrics.get("AP") if isinstance(grape_metrics, dict) else None,
            bbox_metrics[0] if len(bbox_metrics) > 0 else float("nan"),
        )
        grape_ap50 = self._safe_float(
            grape_metrics.get("AP50") if isinstance(grape_metrics, dict) else None,
            bbox_metrics[1] if len(bbox_metrics) > 1 else float("nan"),
        )
        grape_ar100 = self._safe_float(
            grape_metrics.get("AR100") if isinstance(grape_metrics, dict) else None,
            bbox_metrics[8] if len(bbox_metrics) > 8 else float("nan"),
        )

        point_metrics = test_stats.get("grape_point_metrics", {})
        point_pair_count = max(int(point_metrics.get("point_pair_count", 0) or 0), 0)
        has_precision = self._safe_float(point_metrics.get("has_picking_precision"))
        has_recall = self._safe_float(point_metrics.get("has_picking_recall"))
        has_f1 = self._safe_float(point_metrics.get("has_picking_f1"))
        point_mae_x = self._safe_float(point_metrics.get("point_mae_x_px"))
        point_mae_y = self._safe_float(point_metrics.get("point_mae_y_px"))
        point_l2 = self._safe_float(point_metrics.get("point_mean_l2_px"))
        normalized_l2 = point_l2 / point_error_norm_px if math.isfinite(point_l2) else float("nan")

        composite = float("nan")
        if math.isfinite(grape_ap) and math.isfinite(has_f1) and math.isfinite(normalized_l2):
            composite = grape_ap + alpha * has_f1 - beta * normalized_l2

        return {
            "valid_grape_AP": grape_ap,
            "valid_grape_AP50": grape_ap50,
            "valid_grape_AR100": grape_ar100,
            "valid_has_picking_precision": has_precision,
            "valid_has_picking_recall": has_recall,
            "valid_has_picking_F1": has_f1,
            "valid_point_pair_count": point_pair_count,
            "valid_point_MAE_x_px": point_mae_x,
            "valid_point_MAE_y_px": point_mae_y,
            "valid_point_mean_L2_px": point_l2,
            "normalized_point_mean_L2_px": normalized_l2,
            "composite_score": composite,
        }

    def _update_named_point_checkpoints(self, epoch, metrics):
        if not self.output_dir or not self._point_checkpoint_enabled():
            return

        comparisons = {
            "best_grape_ap": ("valid_grape_AP", "max", self.output_dir / "best_grape_ap.pth"),
            "best_has_picking_f1": ("valid_has_picking_F1", "max", self.output_dir / "best_has_picking_f1.pth"),
            "best_point_l2": ("valid_point_mean_L2_px", "min", self.output_dir / "best_point_l2.pth"),
            "best_composite": ("composite_score", "max", self.output_dir / "best_composite.pth"),
        }

        changed = False
        for state_key, (metric_key, mode, path) in comparisons.items():
            if state_key == "best_point_l2" and not self._has_valid_point_l2_candidate(metrics):
                continue
            value = self._safe_float(metrics.get(metric_key))
            if not math.isfinite(value):
                continue
            current = self.point_checkpoint_state.get(state_key, {"epoch": -1, "score": float("-inf"), "metrics": {}})
            if state_key == "best_point_l2" and not self._has_valid_point_l2_candidate(current.get("metrics", {})):
                current_score = float("inf")
            else:
                current_score = self._safe_float(current.get("score"))
            is_better = value < current_score if mode == "min" else value > current_score
            if not math.isfinite(current_score):
                is_better = True
            if not is_better:
                continue
            self.point_checkpoint_state[state_key] = {
                "epoch": int(epoch),
                "score": float(value),
                "metrics": dict(metrics),
                "path": path.name,
            }
            dist_utils.save_on_master(self.state_dict(), path)
            changed = True

        if changed and dist_utils.is_main_process():
            self._save_point_checkpoint_state()

    def _apply_point_phase2_schedule(self, epoch):
        phase2_cfg = self._phase2_cfg()
        if not phase2_cfg or not getattr(self.criterion, "base_weight_dict", None):
            return

        enabled = bool(phase2_cfg.get("enabled", False))
        start_epoch = int(
            phase2_cfg.get("start_epoch", getattr(self.train_dataloader.collate_fn, "stop_epoch", 0))
        )
        multipliers = phase2_cfg.get("weight_multipliers", {}) if enabled and epoch >= start_epoch else {}
        new_weights = dict(self.criterion.base_weight_dict)

        if isinstance(multipliers, dict):
            for key, value in multipliers.items():
                if key in new_weights:
                    new_weights[key] = self._safe_float(self.criterion.base_weight_dict[key], 0.0) * self._safe_float(value, 1.0)

        self.criterion.weight_dict.update(new_weights)

        if self.writer and dist_utils.is_main_process():
            for key, value in new_weights.items():
                self.writer.add_scalar(f"LossWeight/{key}", float(value), epoch)

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')
        self.point_checkpoint_state = self._load_point_checkpoint_state() if self._point_checkpoint_enabled() else {}

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                if isinstance(test_stats[k], dict):
                    continue
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')
            if self._point_checkpoint_enabled() and dist_utils.is_main_process():
                resume_metrics = self._extract_point_eval_metrics(test_stats)
                for state_key, metric_key in (
                    ("best_grape_ap", "valid_grape_AP"),
                    ("best_has_picking_f1", "valid_has_picking_F1"),
                    ("best_point_l2", "valid_point_mean_L2_px"),
                    ("best_composite", "composite_score"),
                ):
                    if state_key == "best_point_l2" and not self._has_valid_point_l2_candidate(resume_metrics):
                        continue
                    existing = self.point_checkpoint_state.get(state_key, {})
                    if int(existing.get("epoch", -1)) >= 0:
                        continue
                    value = self._safe_float(resume_metrics.get(metric_key))
                    if not math.isfinite(value):
                        continue
                    self.point_checkpoint_state[state_key] = {
                        "epoch": int(self.last_epoch),
                        "score": float(value),
                        "metrics": dict(resume_metrics),
                    }
                self._save_point_checkpoint_state()

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            self._apply_point_phase2_schedule(epoch)
            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # Keep periodic snapshots only in stage1 to avoid clutter,
                # but always refresh last.pth so resume continues from the true latest epoch.
                if epoch < self.train_dataloader.collate_fn.stop_epoch and (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            point_eval_metrics = self._extract_point_eval_metrics(test_stats)

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    if isinstance(test_stats[k], dict):
                        for class_name, class_metrics in test_stats[k].items():
                            if isinstance(class_metrics, dict):
                                for metric_name, metric_value in class_metrics.items():
                                    if metric_name == 'category_id':
                                        continue
                                    self.writer.add_scalar(
                                        f'TestPerClass/{class_name}_{metric_name}',
                                        float(metric_value),
                                        epoch,
                                    )
                            else:
                                self.writer.add_scalar(
                                    f'Test/{k}_{class_name}',
                                    float(class_metrics),
                                    epoch,
                                )
                    else:
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if isinstance(test_stats[k], dict):
                    continue

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            if self.writer and dist_utils.is_main_process():
                for metric_name, metric_value in point_eval_metrics.items():
                    if math.isfinite(self._safe_float(metric_value)):
                        self.writer.add_scalar(f'PointValid/{metric_name}', float(metric_value), epoch)

            if self._point_checkpoint_enabled() and dist_utils.is_main_process():
                self._update_named_point_checkpoints(epoch, point_eval_metrics)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'test_point_checkpoint_metrics': point_eval_metrics,
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state
