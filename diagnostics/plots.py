import matplotlib
matplotlib.use("Qt5Agg")

import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style


class TempestPlotter:
    """Automated validation and convergence plotting for the TEMPEST pipeline."""

    LOG_FLOOR = 1e-16
    STYLE = (
        "seaborn-v0_8-darkgrid"
        if "seaborn-v0_8-darkgrid" in plt.style.available
        else "default"
    )

    CONVERGENCE_METRICS = {
        "avg_l2": {
            "label": r"Mean $L_2$ Error",
            "filename": "mean_l2",
            "csv_column": "Avg L2 Error",
        },
        "final_l2": {
            "label": r"Final $L_2$ Error",
            "filename": "final_l2",
            "csv_column": "L2 Error",
        },
        "avg_l1": {
            "label": r"Mean $L_1$ Error",
            "filename": "mean_l1",
            "csv_column": "Avg L1 Error",
        },
        "final_l1": {
            "label": r"Final $L_1$ Error",
            "filename": "final_l1",
            "csv_column": "L1 Error",
        },
    }

    def __init__(self, output_dir="Results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def _safe_log(cls, values):
        arr = np.asarray(values, dtype=float)
        return np.log(np.maximum(arr, cls.LOG_FLOOR))

    @classmethod
    def _floor_errors(cls, values):
        return np.maximum(np.asarray(values, dtype=float), cls.LOG_FLOOR)

    @staticmethod
    def _deduplicate_by_dx(dx_values, error_values):
        """
        Keep the last error recorded for each unique dx.
        Prevents corrupted regressions when the pipeline is re-run or dt varies.
        """
        unique = {}
        for dx, err in zip(dx_values, error_values):
            unique[float(dx)] = float(err)

        sorted_pairs = sorted(unique.items(), key=lambda pair: pair[0], reverse=True)
        dx_out = np.array([pair[0] for pair in sorted_pairs], dtype=float)
        err_out = np.array([pair[1] for pair in sorted_pairs], dtype=float)
        return dx_out, err_out

    @staticmethod
    def _prepare_validation_frame(time_history_df, min_time=0.0):
        """Remove duplicate frame-0 samples and zero errors that break log axes."""
        df = time_history_df.copy()
        df = df[df["time"] > min_time]
        if df.empty:
            return None

        for col in ("l2_error", "l1_error", "max_error"):
            df[col] = TempestPlotter._floor_errors(df[col].values)

        # FuncAnimation with blit=True can record frame 0 multiple times.
        df = df.groupby("time", as_index=False).last()
        return df.sort_values("time")

    def plot_validation(
        self,
        time_history_df,
        eq_name,
        solver_name,
        run_id="transient_errors",
        *,
        N=None,
        dx=None,
        dt=None,
        min_time=0.0,
        show_max_error=False,
    ):
        """
        Transient validation chart: global errors vs physical time.

        Expects columns: time, l2_error, l1_error, max_error.
        """
        df = self._prepare_validation_frame(time_history_df, min_time=min_time)
        if df is None or len(df) == 0:
            print(
                f"Skipping validation plot for {eq_name}: "
                "no post-initialization samples available."
            )
            return None

        t = df["time"].values
        l2_err = df["l2_error"].values
        l1_err = df["l1_error"].values
        max_err = df["max_error"].values

        run_label_parts = [f"N={N}"] if N is not None else []
        if dx is not None:
            run_label_parts.append(f"dx={dx}")
        if dt is not None:
            run_label_parts.append(f"dt={dt}")
        run_label = " | ".join(run_label_parts)

        with style.context(self.STYLE):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

            ax.plot(t, l2_err, color="#0077bb", lw=2, label=r"Global $L_2$ Error")
            ax.plot(
                t,
                l1_err,
                color="#cc3311",
                lw=1.5,
                linestyle="--",
                label=r"Global $L_1$ Error",
            )
            if show_max_error:
                ax.plot(
                    t,
                    max_err,
                    color="#33aa33",
                    lw=1.5,
                    linestyle=":",
                    label=r"Max Pointwise Error",
                )

            title = f"Transient Error: {eq_name} ({solver_name})"
            if run_label:
                title += f"\n{run_label}"
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Elapsed Physical Time", fontsize=10)
            ax.set_ylabel("Absolute Numerical Error", fontsize=10)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", frameon=True)

            save_path = os.path.join(self.output_dir, f"validation_{run_id}.png")
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

        print(f"Validation plot archived: {save_path}")
        return save_path

    def plot_convergence(
        self,
        dx_values,
        error_values,
        eq_name,
        expected_order=2,
        *,
        error_metric="avg_l2",
        deduplicate=True,
        save_sidecar=True,
    ):
        """
        Log-log convergence study with linear regression.

        Parameters
        ----------
        dx_values : array-like
            Grid spacings (coarse to fine after internal sorting/dedup).
        error_values : array-like
            Error metric at each grid (must match the chosen error_metric).
        error_metric : str
            One of avg_l2, final_l2, avg_l1, final_l1 — used for labels/filenames.
        """
        if error_metric not in self.CONVERGENCE_METRICS:
            raise ValueError(
                f"Unknown error_metric '{error_metric}'. "
                f"Choose from: {list(self.CONVERGENCE_METRICS)}"
            )

        dx_arr = np.asarray(dx_values, dtype=float)
        err_arr = np.asarray(error_values, dtype=float)

        if len(dx_arr) != len(err_arr):
            raise ValueError("dx_values and error_values must have the same length.")

        if len(dx_arr) < 2:
            print(
                f"Skipping convergence plot for {eq_name} ({error_metric}): "
                "need at least two distinct grid spacings."
            )
            return None

        if deduplicate:
            dx_arr, err_arr = self._deduplicate_by_dx(dx_arr, err_arr)

        if len(dx_arr) < 2:
            print(
                f"Skipping convergence plot for {eq_name} ({error_metric}): "
                "fewer than two unique dx values after deduplication."
            )
            return None

        # Coarse -> fine for readable log-log trends.
        order = np.argsort(dx_arr)[::-1]
        dx_arr = dx_arr[order]
        err_arr = err_arr[order]

        log_dx = self._safe_log(dx_arr)
        log_err = self._safe_log(err_arr)
        slope, intercept = np.polyfit(log_dx, log_err, 1)

        fit_line = np.exp(intercept) * (dx_arr ** slope)
        theoretical_line = np.exp(intercept) * (dx_arr ** expected_order)

        metric_info = self.CONVERGENCE_METRICS[error_metric]
        metric_label = metric_info["label"]
        metric_slug = metric_info["filename"]

        with style.context(self.STYLE):
            fig, ax = plt.subplots(figsize=(7.5, 6), dpi=100)

            ax.loglog(
                dx_arr,
                err_arr,
                "o",
                color="#0077bb",
                markersize=8,
                markeredgecolor="black",
                label="Measured Error",
            )
            ax.loglog(
                dx_arr,
                fit_line,
                color="#cc3311",
                lw=2,
                label=f"Linear Fit (Observed Order = {slope:.2f})",
            )
            ax.loglog(
                dx_arr,
                theoretical_line,
                linestyle=":",
                color="#888888",
                lw=1.5,
                label=rf"Reference $O(\Delta x^{{{expected_order}}})$",
            )

            ax.set_title(
                f"Grid Convergence: {eq_name}\n({metric_label})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel(r"Grid Spacing $\Delta x$", fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", frameon=True)

            text_str = (
                f"Observed Order: {slope:.2f}\n"
                f"Target Order: {expected_order:.2f}\n"
                f"Points: {len(dx_arr)}"
            )
            ax.text(
                0.05,
                0.05,
                text_str,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
            )

            save_path = os.path.join(
                self.output_dir,
                f"convergence_{eq_name}_{metric_slug}.png",
            )
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

        if save_sidecar:
            self._save_convergence_sidecar(
                save_path,
                dx_arr,
                err_arr,
                slope,
                intercept,
                expected_order,
                error_metric,
            )

        print(f"Convergence plot archived: {save_path}")
        return slope

    def plot_convergence_suite(
        self,
        dx_values,
        metrics_map,
        eq_name,
        expected_order=2,
        *,
        metrics=None,
    ):
        """
        Generate convergence plots for multiple error metrics on the same dx sweep.

        metrics_map : dict
            Keys are metric names (avg_l2, avg_l1, ...), values are error arrays.
        """
        if metrics is None:
            metrics = ["avg_l2", "avg_l1"]

        slopes = {}
        for metric_name in metrics:
            if metric_name not in metrics_map:
                continue
            slope = self.plot_convergence(
                dx_values=dx_values,
                error_values=metrics_map[metric_name],
                eq_name=eq_name,
                expected_order=expected_order,
                error_metric=metric_name,
            )
            if slope is not None:
                slopes[metric_name] = slope
        return slopes

    def _save_convergence_sidecar(
        self,
        plot_path,
        dx_arr,
        err_arr,
        slope,
        intercept,
        expected_order,
        error_metric,
    ):
        """Write regression data next to the PNG for reproducible analysis."""
        sidecar_path = plot_path.replace(".png", ".csv")
        with open(sidecar_path, "w", newline="") as sidecar_file:
            writer = csv.writer(sidecar_file)
            writer.writerow(
                [
                    "dx",
                    "error",
                    "log_dx",
                    "log_error",
                    "observed_order",
                    "expected_order",
                    "log_intercept",
                    "error_metric",
                ]
            )
            for dx, err in zip(dx_arr, err_arr):
                writer.writerow(
                    [
                        dx,
                        err,
                        self._safe_log([dx])[0],
                        self._safe_log([err])[0],
                        slope,
                        expected_order,
                        intercept,
                        error_metric,
                    ]
                )
        print(f"Convergence data archived: {sidecar_path}")

    @classmethod
    def from_metrics_rows(
        cls,
        rows,
        output_dir,
        eq_name,
        expected_order=2,
        *,
        metrics=None,
        group_filter=None,
    ):
        """
        Build convergence plots directly from master_metrics-style row dicts.

        group_filter : callable(row) -> bool, optional
            Restrict which rows are included (e.g. same equation/operator).
        """
        if metrics is None:
            metrics = ["avg_l2", "avg_l1"]

        if group_filter is not None:
            rows = [row for row in rows if group_filter(row)]

        if not rows:
            print(f"No metrics rows matched filter for {eq_name}.")
            return {}

        dx_values = [float(row["DX"]) for row in rows]
        metrics_map = {}
        for metric_name in metrics:
            csv_col = cls.CONVERGENCE_METRICS[metric_name]["csv_column"]
            metrics_map[metric_name] = [float(row[csv_col]) for row in rows]

        plotter = cls(output_dir=output_dir)
        return plotter.plot_convergence_suite(
            dx_values=dx_values,
            metrics_map=metrics_map,
            eq_name=eq_name,
            expected_order=expected_order,
            metrics=metrics,
        )
