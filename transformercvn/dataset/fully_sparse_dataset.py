from typing import Tuple, List

import numba
import numpy as np
import torch
from torch import Tensor
# noinspection PyUnresolvedReferences
from MinkowskiEngine import to_sparse

from transformercvn.dataset.sparse_prong_pixel_target_dataset import SparseProngPixelTargetDataset


@numba.njit()
def sparse_to_sparse(coords, values, num_prongs, shape_scale):
    index_mapping = dict()
    output_coordinates = np.zeros((values.shape[0] + num_prongs, 3), dtype=np.int32)
    output_values = np.zeros((values.shape[0] + num_prongs, cvnmap_shape[2]), dtype=np.float32)

    num_unique_indices = 0

    for i in range(values.shape[0]):
        v = values[i]
        if v == 0:
            continue

        b = coords[0, i]
        c = coords[1, i]
        y = coords[2, i]
        x = coords[3, i]

        flat_index = shape_scale[0] * b + shape_scale[1] * y + shape_scale[2] * x

        if flat_index not in index_mapping:
            output_index = num_unique_indices
            index_mapping[flat_index] = num_unique_indices
            num_unique_indices += 1
        else:
            output_index = index_mapping[flat_index]

        output_coordinates[output_index][0] = b
        output_coordinates[output_index][1] = y
        output_coordinates[output_index][2] = x

        output_values[output_index][c] = v

    non_zero_planes = set(output_coordinates[:, 0])
    all_planes = set(np.arange(num_prongs, dtype=np.int32))
    missing_planes = all_planes.difference(non_zero_planes)

    for i in missing_planes:
        output_coordinates[num_unique_indices, 0] = i
        num_unique_indices += 1

    output_coordinates = output_coordinates[:num_unique_indices]
    output_values = output_values[:num_unique_indices]

    sorting_indices = np.argsort((output_coordinates * shape_scale).sum(1))
    output_coordinates = np.ascontiguousarray(output_coordinates[sorting_indices])
    output_values = np.ascontiguousarray(output_values[sorting_indices])

    return output_coordinates, output_values


class SparseCollation:
    def __init__(self):
        pass

    @staticmethod
    def collate_sparse(coordinates: List[Tensor], values: List[Tensor], masks: List[Tensor]) -> Tuple[Tensor, Tensor]:
        offset = coordinates[0].new_zeros([1, coordinates[0].shape[1]])
        offset[0, 0] = 1

        total_prongs = 0
        collated_coordinates = []
        for coord, num_prongs in zip(coordinates, masks):
            collated_coordinates.append(coord + offset * total_prongs)
            total_prongs += num_prongs.sum().item()

        collated_coordinates = torch.cat(collated_coordinates)
        collated_values = torch.cat(values)

        return collated_coordinates, collated_values

    def __call__(self, list_data):
        (
            features,
            extra,
            event_coordinates,
            event_values,
            event_masks,
            prong_coordinates,
            prong_values,
            prong_masks,
            event_targets,
            prong_targets
        ) = list(zip(*list_data))

        features = torch.stack(features)
        extra = torch.stack(extra)

        event_coordinates, event_values = self.collate_sparse(event_coordinates, event_values, event_masks)
        prong_coordinates, prong_values = self.collate_sparse(prong_coordinates, prong_values, prong_masks)

        event_masks = torch.stack(event_masks)
        prong_masks = torch.stack(prong_masks)

        event_targets = torch.stack(event_targets)
        prong_targets = torch.stack(prong_targets)

        return (
            features,
            extra,
            event_coordinates,
            event_values,
            event_masks,
            prong_coordinates,
            prong_values,
            prong_masks,
            event_targets,
            prong_targets
        )


class FullySparseDataset(SparseProngPixelTargetDataset):
    def __init__(self, data_file: str, limit_index=1.0):
        super(FullySparseDataset, self).__init__(data_file, limit_index)

        self.shape_scale = torch.from_numpy(np.ascontiguousarray(np.cumprod((*self.pixel_shape, 1)[::-1])[::-1]))

    def dense_to_sparse(self, pixels: Tensor) -> Tuple[Tensor, Tensor]:
        num_prongs = pixels.shape[0]
        pixel_features = pixels.shape[1]

        # Hack to flip one of the image planes of DUNE dataset but not NOVA dataset
        if pixel_features == 3:
            pixels[:, 1, :, :] = torch.flip(pixels[:, 1, :, :], (1,))

        # Convert to basic sparse tensor
        pixels = to_sparse(pixels)

        # Extract the two sparse tensors
        coordinates = pixels.coordinates
        features = pixels.features

        new_coordinates = []
        new_features = []

        # Add dummy zero pixels in the center of all prongs to ensure consistent shape
        non_zero_planes = set(torch.unique(coordinates[:, 0]).tolist())
        for i in set(range(num_prongs)).difference(non_zero_planes):
            new_features.append(features.new_zeros(1, features.shape[1]))
            new_coordinates.append(coordinates.new_zeros(1, coordinates.shape[1]))
            new_coordinates[-1][0, 0] = i

        coordinates = torch.cat((coordinates, *new_coordinates))
        features = torch.cat((features, *new_features))

        # Sort coordinates by their prong index for faster computation
        flat_coordinates = (coordinates * self.shape_scale).sum(1)
        sorting_indices = torch.sort(flat_coordinates).indices

        coordinates = coordinates[sorting_indices].contiguous()
        features = features[sorting_indices].contiguous()

        return coordinates, features

    def __getitem__(self, item) -> Tuple[Tensor, ...]:
        mask = self.mask[item]
        num_prongs = mask.sum().item()

        pixels = self.prong_pixels.get_compressed(item, num_prongs)
        pixels = pixels.reshape(num_prongs, self.pixel_features, *self.pixel_shape)
        # C = to_sparse(pixels).C
        # F = to_sparse(pixels).F
        # S = self.prong_pixels.get_sparse_data(item)

        event_pixels = pixels[:1]
        prong_pixels = pixels[1:]
        event_mask = mask[:1]
        prong_mask = mask[1:]

        event_coordinates, event_values = self.dense_to_sparse(event_pixels)
        prong_coordinates, prong_values = self.dense_to_sparse(prong_pixels)

        features = self.data[item][1:]
        extra = self.extra[item]

        targets = self.targets[item]
        prong_targets = self.prong_targets[item][1:]

        return (
            features,
            extra,
            event_coordinates,
            event_values,
            event_mask,
            prong_coordinates,
            prong_values,
            prong_mask,
            targets,
            prong_targets
        )
