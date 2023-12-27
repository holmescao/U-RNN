from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error


class ModelEvaluation:
    def __init__(self, pred, label,  pred_cls_h, label_cls_h, 
                 unit=1000, top_k=100, patch_size=5, figs_dir="figs/maxae-patch"):
        self.pred = pred / unit
        self.label = label / unit
        self.pred_cls_h = pred_cls_h 
        self.label_cls_h = label_cls_h 
        
        self.top_k = top_k
        self.patch_size = patch_size
        self.figs_dir = figs_dir

    @staticmethod
    def cal_max_AE(pred, label):
        error = pred - label
        ae = np.abs(error)
        return np.max(ae)

    @staticmethod
    def cal_MSE(pred, label):
        return mean_squared_error(label, pred)
    
    @staticmethod
    def cal_RMSE(pred, label):
        return np.sqrt(mean_squared_error(label, pred))

    @staticmethod
    def cal_MAE(pred, label):
        return np.mean(np.abs(pred - label))

    def visualize_max_errors(self, pred_h, label_h, pred_cls_h, label_cls_h, top_k=50, patch_size=5, figs_dir="/mnt/data/figs/maxae-patch"):
        # Calculate the absolute error
        ae = np.abs(pred_h - label_h)

        # Flatten the error array and get the indices of the top values
        flattened_ae = ae.flatten()  # Ensuring that we flatten the array
        top_indices = np.argpartition(flattened_ae, -top_k)[-top_k:]
        top_indices_sorted = top_indices[np.argsort(-flattened_ae[top_indices])]

        # Convert flat indices to 3D indices
        top_coords = np.unravel_index(top_indices_sorted, ae.shape)

        # Create the directory for figures if it doesn't exist
        os.makedirs(figs_dir, exist_ok=True)

        # Define the size of the patch
        half_n = patch_size // 2
        
        # Find the common min and max for the shared color scale
        vmin = np.min([pred_h.min(), label_h.min(), -ae.max()])
        vmax = np.max([pred_h.max(), label_h.max(), ae.max()])
        vmax = np.max([np.abs(vmin),vmax])
        vmin = - vmax

        vmax_cls = 1
        vmin_cls = -1
        # Iterate over the top indices
        for idx, coord in enumerate(zip(*top_coords)):
            z, x, y = coord

            # Create patches for pred_h, label_h, and their difference
            pred_patch = pred_h[z, max(0, x-half_n):min(pred_h.shape[1], x+half_n+1), max(0, y-half_n):min(pred_h.shape[2], y+half_n+1)]
            label_patch = label_h[z, max(0, x-half_n):min(label_h.shape[1], x+half_n+1), max(0, y-half_n):min(label_h.shape[2], y+half_n+1)]
            diff_patch = pred_patch - label_patch

            # 新增
            pred_cls_patch = pred_cls_h[z, max(0, x-half_n):min(pred_cls_h.shape[1], x+half_n+1), max(0, y-half_n):min(pred_cls_h.shape[2], y+half_n+1)]
            label_cls_patch = label_cls_h[z, max(0, x-half_n):min(label_cls_h.shape[1], x+half_n+1), max(0, y-half_n):min(label_cls_h.shape[2], y+half_n+1)]
            diff_cls_patch = pred_cls_patch - label_cls_patch


            # Plotting patches
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))

            # Titles for subplots
            titles = ['Prediction reg Patch', 'Label reg Patch', 'Difference reg Patch', 'Time Series reg']

            # Plot the time series on the last subplot of the first row
            time_series = np.arange(pred_h.shape[0])
            axs[0, -1].plot(time_series, pred_h[:, x, y], label='Prediction')
            axs[0, -1].plot(time_series, label_h[:, x, y], label='Label')
            axs[0, -1].set_title(titles[-1])
            axs[0, -1].legend()

            # Plot the patches for the first row
            patches = [pred_patch, label_patch, diff_patch]
            for ax, patch, title in zip(axs[0, :3], patches, titles[:3]):
                im = ax.imshow(patch, cmap='bwr', vmin=vmin, vmax=vmax)
                # Add text annotations
                for i in range(patch.shape[0]):
                    for j in range(patch.shape[1]):
                        text_color = 'w' if np.abs(patch[i, j]) > (vmax - vmin) / 2 else 'k'
                        ax.text(j, i, '{:.2f}'.format(patch[i, j]), ha='center', va='center', color=text_color)
                ax.set_title(title)
                ax.axis('off')

            # Titles for the second row subplots
            titles_cls = ['Prediction cls Patch', 'Label cls Patch', 'Difference cls Patch', 'Time Series cls']

            # Plot the time series on the last subplot of the second row
            axs[1, -1].plot(time_series, pred_cls_h[:, x, y], label='Prediction cls')
            axs[1, -1].plot(time_series, label_cls_h[:, x, y], label='Label cls')
            axs[1, -1].set_title(titles_cls[-1])
            axs[1, -1].legend()

            # Plot the patches for the second row
            patches_cls = [pred_cls_patch, label_cls_patch, diff_cls_patch]
            for ax, patch, title in zip(axs[1, :3], patches_cls, titles_cls[:3]):
                im_cls = ax.imshow(patch, cmap='bwr', vmin=vmin_cls, vmax=vmax_cls)
                # Add text annotations
                for i in range(patch.shape[0]):
                    for j in range(patch.shape[1]):
                        text_color = 'w' if np.abs(patch[i, j]) > (vmax_cls - vmin_cls) / 2 else 'k'
                        ax.text(j, i, '{:.2f}'.format(patch[i, j]), ha='center', va='center', color=text_color)
                ax.set_title(title)
                ax.axis('off')


            # Shared colorbar for the patches
            fig.colorbar(im, ax=axs[0, :3], fraction=0.046, pad=0.04)

            # Shared colorbar for the patches in the second row
            fig.colorbar(im_cls, ax=axs[1, :3], fraction=0.046, pad=0.04)

            # Save the figure
            fig_path = os.path.join(figs_dir, f"combined_{idx}.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)

            # Indicate progress
            print(f"Saved: {fig_path}")

    def stats(self,thred = 1):
        # count
        
        ae = np.abs(self.pred - self.label)
        count = np.sum(ae >= thred)
        print(f"ae > {thred}: {count}")
        # import sys
        # sys.exit()
        
    def evaluate(self):
        pred_flat, label_flat = self.pred.flatten(), self.label.flatten()
        # mse = self.cal_MSE(pred_flat, label_flat)
        rmse = self.cal_RMSE(pred_flat, label_flat)
        mae = self.cal_MAE(pred_flat, label_flat)
        # maxae = self.cal_max_AE(pred_flat, label_flat)
        # self.visualize_max_errors(self.pred, self.label, 
        #                           self.pred_cls_h,self.label_cls_h,
        #                           top_k=self.top_k, patch_size=self.patch_size, 
        #                           figs_dir=self.figs_dir)

        return rmse, mae
        # return mse,rmse, mae, maxae


if __name__ == "__main__":
    exp_dir = "../../exp/"
    best = "20231216_235718_449685"

    best_exp_dir = os.path.join(exp_dir, best, "save_train_loss")
    key = "h"
    sample_id = 0

    

    folder_path = best_exp_dir
    thred_list = [0.5]
    # thred_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for thred in thred_list:
        all_mse = []
        all_rmse = []
        all_mae = []
        all_maxae = []
        bar = tqdm(range(len(os.listdir(folder_path))//2))
        for x in bar:  # loop from 0 to 7
            thred_str= str(thred).replace(".","")
            label_path = f"{folder_path}/label_{x}_{thred_str}.npz"
            pred_path = f"{folder_path}/pred_{x}_{thred_str}.npz"

            label_data = np.load(label_path)
            pred_data = np.load(pred_path)

            label_h = label_data["label_h"][0].squeeze(1)
            label_cls_h = label_data["label_cls_h"][0].squeeze(1)
            pred_h = pred_data["pred_h"][0].squeeze(1)
            pred_cls_h = pred_data["pred_cls_h"][0].squeeze(1)
            

            evaluator = ModelEvaluation(pred_h, label_h, pred_cls_h, label_cls_h,
                                        top_k=1,
                                        figs_dir=f"figs/thred@{thred_str}_maxae-patch")
            # for thred in range(1, 10):
            #     evaluator.stats(thred*0.1)
            
            # mse, rmse, mae, maxae = evaluator.evaluate()
            rmse, mae = evaluator.evaluate()

            all_mae.append(mae)
            # all_mse.append(mse)
            all_rmse.append(rmse)
            # all_maxae.append(maxae)

        # 分割实测降雨和设计降雨的数据
        real_design = 7
        # real_design = 29
        real_rainfall_mae = all_mae[:real_design]
        design_rainfall_mae = all_mae[real_design:]

        # real_rainfall_mse = all_mse[:real_design]
        # design_rainfall_mse = all_mse[real_design:]

        real_rainfall_rmse = all_rmse[:real_design]
        design_rainfall_rmse = all_rmse[real_design:]

        # real_rainfall_maxae = all_maxae[:real_design]
        # design_rainfall_maxae = all_maxae[real_design:]

        # 计算实测降雨的指标
        mean_real_mae = np.mean(real_rainfall_mae)
        std_real_mae = np.std(real_rainfall_mae)

        # mean_real_mse = np.mean(real_rainfall_mse)
        # std_real_mse = np.std(real_rainfall_mse)

        mean_real_rmse = np.mean(real_rainfall_rmse)
        std_real_rmse = np.std(real_rainfall_rmse)

        # mean_real_maxae = np.mean(real_rainfall_maxae)
        # std_real_maxae = np.std(real_rainfall_maxae)

        # 计算设计降雨的指标
        mean_design_mae = np.mean(design_rainfall_mae)
        std_design_mae = np.std(design_rainfall_mae)

        # mean_design_mse = np.mean(design_rainfall_mse)
        # std_design_mse = np.std(design_rainfall_mse)

        mean_design_rmse = np.mean(design_rainfall_rmse)
        std_design_rmse = np.std(design_rainfall_rmse)

        # mean_design_maxae = np.mean(design_rainfall_maxae)
        # std_design_maxae = np.std(design_rainfall_maxae)

        # 打印结果
        print("实测降雨指标：")
        print(f"MAE. mean:{mean_real_mae} std:{std_real_mae}")
        # print(f"MSE. mean:{mean_real_mse} std:{std_real_mse}")
        print(f"RMSE. mean:{mean_real_rmse} std:{std_real_rmse}")
        # print(f"MAXAE. mean:{mean_real_maxae} std:{std_real_maxae}")

        print("\n设计降雨指标：")
        print(f"MAE. mean:{mean_design_mae} std:{std_design_mae}")
        # print(f"MSE. mean:{mean_design_mse} std:{std_design_mse}")
        print(f"RMSE. mean:{mean_design_rmse} std:{std_design_rmse}")
        # print(f"MAXAE. mean:{mean_design_maxae} std:{std_design_maxae}")

        
        print("\n所有降雨指标：")
        mean_mae = np.mean(all_mae)
        std_mae = np.std(all_mae)
        mean_rmse = np.mean(all_rmse)
        std_rmse = np.std(all_rmse)
        # mean_maxae = np.mean(all_maxae)
        # std_maxae = np.std(all_maxae)
        print(f"MAE. mean:{mean_mae} std:{std_mae}")
        print(f"RMSE. mean:{mean_rmse} std:{std_rmse}")
        # print(f"MAXAE. mean:{mean_maxae} std:{std_maxae}")