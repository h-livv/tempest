import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

class ValidationRendererRegistry:
    _renderers = []
    
    @classmethod
    def register(cls, renderer):
        cls._renderers.append(renderer)
        return renderer
        
    @classmethod
    def resolve(cls, grid):
        if grid is None:
            return None
        for r in cls._renderers:
            if r.can_handle(grid):
                return r
        return None

@ValidationRendererRegistry.register
class Scalar1DValidationRenderer:
    @staticmethod
    def can_handle(grid):
        return grid.ndim == 1
        
    @staticmethod
    def render(plotter, grid, u_numerical, u_analytical, eq_name, solver_name, run_id, time_history_df, raw_tensor_data):
        x = grid.coordinates[0]
        num_y = u_numerical[0] if u_numerical.ndim > 1 else u_numerical
        anal_y = u_analytical[0] if u_analytical.ndim > 1 else u_analytical

        with style.context(plotter.STYLE):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
            ax.plot(x, anal_y, color="#cc3311", lw=2.5, label="Analytical Solution")
            ax.plot(x, num_y, color="#0077bb", lw=1.5, linestyle="--", label="Numerical Solution")
            ax.set_title(f"State Comparison: {eq_name} ({solver_name})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Space (x)", fontsize=10)
            ax.set_ylabel("State (u)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", frameon=True)
            state_save_path = os.path.join(plotter.output_dir, f"state_comparison_{run_id}.png")
            fig.savefig(state_save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"State comparison plot archived: {state_save_path}")

            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
            ax.plot(x, num_y - anal_y, color="#cc3311", lw=2, label="Difference (Numerical - Analytical)")
            ax.set_title(f"State Difference: {eq_name} ({solver_name})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Space (x)", fontsize=10)
            ax.set_ylabel("Difference (u)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", frameon=True)
            diff_save_path = os.path.join(plotter.output_dir, f"state_difference_{run_id}.png")
            fig.savefig(diff_save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"State difference plot archived: {diff_save_path}")

        if raw_tensor_data is not None:
            with style.context(plotter.STYLE):
                fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                data_slice = raw_tensor_data[:, 0, :] if raw_tensor_data.ndim == 3 else raw_tensor_data
                t_vals = time_history_df["time"].values
                T_grid, X_grid = np.meshgrid(t_vals, x, indexing='ij')
                im = ax.pcolormesh(X_grid, T_grid, data_slice, cmap="inferno", shading="auto")
                fig.colorbar(im, ax=ax, label="State Amplitude")
                ax.set_title(f"Spacetime Diagram: {eq_name} ({solver_name})", fontsize=12, fontweight="bold")
                ax.set_xlabel("Space (x)", fontsize=10)
                ax.set_ylabel("Time (t)", fontsize=10)
                spacetime_save_path = os.path.join(plotter.output_dir, f"spacetime_{run_id}.png")
                fig.savefig(spacetime_save_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Spacetime plot archived: {spacetime_save_path}")


@ValidationRendererRegistry.register
class Scalar2DValidationRenderer:
    @staticmethod
    def can_handle(grid):
        return grid.ndim == 2

    @staticmethod
    def render(plotter, grid, u_numerical, u_analytical, eq_name, solver_name, run_id, time_history_df, raw_tensor_data):
        X, Y = grid.coordinates[0], grid.coordinates[1]
        num_y = u_numerical[0] if u_numerical.ndim > 2 else u_numerical
        anal_y = u_analytical[0] if u_analytical.ndim > 2 else u_analytical
        
        vmin = min(np.min(num_y), np.min(anal_y))
        vmax = max(np.max(num_y), np.max(anal_y))
        
        with style.context(plotter.STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
            
            im0 = axes[0].pcolormesh(X, Y, num_y, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
            axes[0].set_title("Numerical Solution")
            fig.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].pcolormesh(X, Y, anal_y, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
            axes[1].set_title("Analytical Solution")
            fig.colorbar(im1, ax=axes[1])
            
            diff = np.abs(num_y - anal_y)
            im2 = axes[2].pcolormesh(X, Y, diff, cmap="inferno", shading="auto")
            axes[2].set_title("Absolute Error")
            fig.colorbar(im2, ax=axes[2])
            
            for ax in axes:
                ax.set_xlabel("Space (x)")
                ax.set_ylabel("Space (y)")
                ax.set_aspect('equal')
                
            fig.suptitle(f"State Comparison: {eq_name} ({solver_name})", fontsize=14, fontweight="bold")
            
            save_path = os.path.join(plotter.output_dir, f"state_comparison_2d_{run_id}.png")
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"2D State comparison plot archived: {save_path}")
