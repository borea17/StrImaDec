import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as transforms


class AIR_BaseClass(pl.LightningModule):

    expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
    norm_target_rectangle = torch.tensor(
        [[-1.0, -1.0, 1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, -1, -1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    ).view(1, 3, 5)

    def __init__(self) -> None:
        super().__init__()
        return

    ########################################
    ############# PLOT FUNCTIONS ###########
    ########################################

    def plot_reconstructions_and_attention_rects(self, images):
        """

        Args:
            images (torch.tensor) shape [batch_size, img_channels, img_dim, img_dim]
        """
        batch_size, img_channels = images.shape[0:2]
        colors_rect = ["red", "green", "yellow", "blue", "magenta"]
        # feed images through model
        results = self.forward(images.to(self.device), self.N_train, True)

        fig = plt.figure(figsize=(8, 3))
        for counter in range(batch_size):
            orig_img = images[counter]
            # plot original data
            ax = plt.subplot(2, batch_size, 1 + counter)
            if img_channels == 1:
                plt.imshow(orig_img[0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                plt.imshow(transforms.ToPILImage()(orig_img))
            plt.axis("off")
            if counter == 0:
                ax.annotate(
                    "Data",
                    xy=(-0.05, 0.5),
                    xycoords="axes fraction",
                    fontsize=14,
                    va="center",
                    ha="right",
                    rotation=90,
                )

            attention_rect = results["attention_rects"][counter]
            x_tilde = torch.clamp(results["x_tilde"][counter], 0, 1)
            # plot reconstruction
            ax = plt.subplot(2, batch_size, 1 + counter + batch_size)
            if img_channels == 1:
                plt.imshow(x_tilde[0].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                plt.imshow(transforms.ToPILImage()(x_tilde))
            plt.axis("off")
            # show attention windows
            for step in range(self.N_train):
                rect = attention_rect[step].detach().cpu().numpy()
                if rect.sum() > 0:  # valid rectangle
                    plt.plot(rect[0], rect[1] - 0.5, color=colors_rect[step])
        return fig

    ########################################
    ### MODEL SPECIFIC HELPER FUNCTIONS ####
    ########################################

    @staticmethod
    def image_to_window(x, z_i_where, img_channels, window_size):
        grid_shape = (z_i_where.shape[0], img_channels, window_size, window_size)
        z_i_where_inv = AIR_BaseClass.invert_z_where(z_i_where)
        x_att_i = AIR_BaseClass.spatial_transform(x, z_i_where_inv, grid_shape)
        return x_att_i

    @staticmethod
    def window_to_image(x_tilde_att_i, z_i_where, img_shape):
        grid_shape = (z_i_where.shape[0], *img_shape)
        x_tilde_i = AIR_BaseClass.spatial_transform(x_tilde_att_i, z_i_where, grid_shape)
        return x_tilde_i

    @staticmethod
    def spatial_transform(x, z_where, grid_shape):
        theta_matrix = AIR_BaseClass.z_where_to_transformation_matrix(z_where)
        grid = F.affine_grid(theta_matrix, grid_shape, align_corners=False)
        out = F.grid_sample(x, grid, align_corners=False)
        return out

    @staticmethod
    def z_where_to_transformation_matrix(z_i_where):
        """taken from
        https://github.com/pyro-ppl/pyro/blob/dev/examples/air/air.py
        """
        batch_size = z_i_where.shape[0]
        out = torch.cat((z_i_where.new_zeros(batch_size, 1), z_i_where), 1)
        ix = AIR_BaseClass.expansion_indices
        if z_i_where.is_cuda:
            ix = ix.cuda()
        out = torch.index_select(out, 1, ix)
        theta_matrix = out.view(batch_size, 2, 3)
        return theta_matrix

    @staticmethod
    def get_attention_rectangle(z_i_where, img_size):
        """transforms a normalized target rectangle into a source rectangle using
        z_i_where and the image size to mimick image-to-window transformation

        Args:
            z_i_where (torch tensor): implicitely describing the transformation
            img_size (int): size of the whole image in one dimension

        Returns:
            source_rectangle (torch tensor): attented rectangle defined in image coordinates
        """
        batch_size = z_i_where.shape[0]
        z_i_where_inv = AIR_BaseClass.invert_z_where(z_i_where)
        theta_matrix = AIR_BaseClass.z_where_to_transformation_matrix(z_i_where_inv)
        target_rectangle = AIR_BaseClass.norm_target_rectangle.expand(batch_size, 3, 5).to(
            z_i_where.device
        )
        source_rectangle_normalized = torch.matmul(theta_matrix, target_rectangle)
        # remap into absolute values
        source_rectangle = 0 + (img_size / 2) * (source_rectangle_normalized + 1)
        return source_rectangle

    @staticmethod
    def invert_z_where(z_where):
        scale = z_where[:, 0:1] + 1e-9

        z_where_inv = torch.zeros_like(z_where)
        z_where_inv[:, 0:1] = 1 / scale
        z_where_inv[:, 1:3] = -z_where[:, 1:3] / scale
        return z_where_inv