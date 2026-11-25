from __future__ import annotations

from src.models.unet3d import ConditionalUNet3D
from src.models.unet3d_grouped import GroupedConditionalUNet3D
from src.models.unet3d_grouped_v2 import GroupedConditionalUNet3DV2
from src.models.unet3d_aerosol_residual import GroupedAerosolResidualUNet3D
from src.models.unet3d_aerosol_residual_st48 import GroupedAerosolResidualST48UNet3D


def is_grouped_b13_model(model_cfg: dict) -> bool:
    name = str(model_cfg.get("architecture", model_cfg.get("name", "conditional_unet3d"))).lower()
    return "grouped" in name or "wrfcloud" in name


def is_grouped_b13_v2_model(model_cfg: dict) -> bool:
    name = str(model_cfg.get("architecture", model_cfg.get("name", ""))).lower()
    return "grouped_v2" in name or "film_spatiotemporal_v2" in name or name.endswith("wrfcloud_v2")


def build_b13_model(
    model_cfg: dict,
    target_channels: int,
    condition_channels: int,
    wrf_cloud_channels: int = 0,
):
    common = {
        "target_channels": int(target_channels),
        "condition_channels": int(condition_channels),
        "base_channels": int(model_cfg.get("base_channels", 16)),
        "channel_mult": list(model_cfg.get("channel_mult", [1, 2, 4])),
        "timestep_embedding": str(model_cfg.get("timestep_embedding", "scalar")),
    }
    architecture = str(model_cfg.get("architecture", model_cfg.get("name", ""))).lower()
    if architecture in {"grouped_aerosol_residual_v1", "grouped_aerosol_residual_st48_v2"}:
        residual_class = (
            GroupedAerosolResidualST48UNet3D
            if architecture == "grouped_aerosol_residual_st48_v2"
            else GroupedAerosolResidualUNet3D
        )
        model = residual_class(
            **common,
            wrf_cloud_channels=int(wrf_cloud_channels),
            condition_embed_channels=int(model_cfg.get("condition_embed_channels", 80)),
            wrf_cloud_embed_channels=int(model_cfg.get("wrf_cloud_embed_channels", 80)),
            aerosol_3d_species=int(model_cfg.get("aerosol_3d_species", 9)),
            aerosol_3d_levels=int(model_cfg.get("aerosol_3d_levels", 6)),
            aerosol_3d_embed_channels=int(model_cfg.get("aerosol_3d_embed_channels", 16)),
            aerosol_3d_spatial_pool=int(model_cfg.get("aerosol_3d_spatial_pool", 1)),
            aerosol_3d_activation_checkpointing=bool(model_cfg.get("aerosol_3d_activation_checkpointing", True)),
            temporal_attention_heads=int(model_cfg.get("temporal_attention_heads", 8)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            residual_enabled=bool(model_cfg.get("residual_enabled", True)),
        )
    elif is_grouped_b13_v2_model(model_cfg):
        model = GroupedConditionalUNet3DV2(
            **common,
            wrf_cloud_channels=int(wrf_cloud_channels),
            condition_embed_channels=int(model_cfg.get("condition_embed_channels", 88)),
            wrf_cloud_embed_channels=int(model_cfg.get("wrf_cloud_embed_channels", 88)),
            time_embed_channels=int(model_cfg.get("time_embed_channels", 256)),
            temporal_attention_heads=int(model_cfg.get("temporal_attention_heads", 8)),
            temporal_attention_chunk_size=int(model_cfg.get("temporal_attention_chunk_size", 2048)),
        )
    elif is_grouped_b13_model(model_cfg):
        model = GroupedConditionalUNet3D(
            **common,
            wrf_cloud_channels=int(wrf_cloud_channels),
            condition_embed_channels=int(model_cfg.get("condition_embed_channels", 80)),
            wrf_cloud_embed_channels=int(model_cfg.get("wrf_cloud_embed_channels", 80)),
            aerosol_3d_species=int(model_cfg.get("aerosol_3d_species", 0)),
            aerosol_3d_levels=int(model_cfg.get("aerosol_3d_levels", 0)),
            aerosol_3d_embed_channels=int(model_cfg.get("aerosol_3d_embed_channels", 32)),
            aerosol_3d_spatial_pool=int(model_cfg.get("aerosol_3d_spatial_pool", 1)),
            aerosol_3d_activation_checkpointing=bool(
                model_cfg.get("aerosol_3d_activation_checkpointing", False)
            ),
            temporal_attention_heads=int(model_cfg.get("temporal_attention_heads", 8)),
        )
    else:
        model = ConditionalUNet3D(**common)
    model.diffusion_steps = int(model_cfg.get("diffusion_steps", 1000))
    return model


def b13_model_kwargs(wrf_cloud_condition, aerosol_3d_condition=None):
    if wrf_cloud_condition is None and aerosol_3d_condition is None:
        return None
    values = {}
    if wrf_cloud_condition is not None:
        values["wrf_cloud_condition"] = wrf_cloud_condition
    if aerosol_3d_condition is not None:
        values["aerosol_3d_condition"] = aerosol_3d_condition
    return values
