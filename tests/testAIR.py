from unittest import TestCase

import numpy as np
import torch

from strimadec.models import AIR


class testAIR(TestCase):

    BATCH_SIZE = 10

    def test_z_where_to_transformation_matrix_works_as_expected(self):
        rnd_z_where = 10 * torch.randn(testAIR.BATCH_SIZE, 3)
        trans_matrix_exp = torch.zeros(testAIR.BATCH_SIZE, 2, 3)
        for i_batch in range(testAIR.BATCH_SIZE):
            s = rnd_z_where[i_batch, 0]
            t_x, t_y = rnd_z_where[i_batch, 1], rnd_z_where[i_batch, 2]
            trans_matrix_exp[i_batch, 0, 0] = s
            trans_matrix_exp[i_batch, 1, 1] = s
            trans_matrix_exp[i_batch, 0, 2] = t_x
            trans_matrix_exp[i_batch, 1, 2] = t_y
        trans_matrix = AIR.z_where_to_transformation_matrix(rnd_z_where)
        self.assertTrue((trans_matrix_exp - trans_matrix).pow(2).sum() < 1e-12)
        return

    def test_get_attention_rectangle_works_as_expected(self):
        # check that identity transformation works
        z_where_no_transform = torch.zeros(testAIR.BATCH_SIZE, 3)
        z_where_no_transform[:, 0] = 1
        norm_source_rectangle_no_transform = (
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0], [0.0, 1.0, 1.0, 0.0, 0.0]])
            .view(1, 2, 5)
            .expand(testAIR.BATCH_SIZE, 2, 5)
        )
        fake_img_size = np.random.randint(10, 100)
        source_rectangle_exp = fake_img_size * norm_source_rectangle_no_transform
        source_rectangle = AIR.get_attention_rectangle(z_where_no_transform, fake_img_size)
        self.assertTrue((source_rectangle_exp - source_rectangle).pow(2).sum() < 1e-12)
        # check that `get_attention_rectangle` works for more dedicated case
        off_set_x = np.random.randint(1, fake_img_size)
        off_set_y = np.random.randint(1, fake_img_size)

        source_rectangle_exp = fake_img_size * (
            torch.tensor(
                [
                    [off_set_x, off_set_x, 1.0 + off_set_x, 1.0 + off_set_x, off_set_x],
                    [off_set_y, 1.0 + off_set_y, 1.0 + off_set_y, off_set_y, off_set_y],
                ]
            )
            .view(1, 2, 5)
            .expand(testAIR.BATCH_SIZE, 2, 5)
        )
        z_where_off_set_x_y = z_where_no_transform
        # add a minus because of inverting z_i_where
        # multiply by two due normalized target rect between -1, 1
        z_where_off_set_x_y[:, 1] = -off_set_x * 2
        z_where_off_set_x_y[:, 2] = -off_set_y * 2
        source_rectangle = AIR.get_attention_rectangle(z_where_off_set_x_y, fake_img_size)
        self.assertTrue((source_rectangle_exp - source_rectangle).pow(2).sum() < 1e-12)
        return