import os
import numpy as np
import rasterio
import warnings
from typing import Dict, List, Any
from qgis.core import QgsProcessingFeedback


class TemporalAnalyticsEngine:
    """
    Multi-Year Coastal Intelligence Engine:
    - StatCD Hypothesis-Tested Bathymetric Change Detection (Z-score significance)
    - Sediment Mass Balance & Volume Change (in m³)
    - Morphological Stability Index Map (MSI)
    """

    def _compute_pair(self, y1, y2, sdb_maps, uncertainty_maps, output_dir, feedback, osw_shp, is_overall=False, all_years=None, overall_trend_method="Long-term Trend", analysis_type="Sequential"):
        from rasterio.warp import reproject, Resampling
        
        statcd_raster_path = os.path.join(output_dir, f"Volumetric_Erosion_Accretion_{y1}_{y2}.tif")
        msi_raster_path = os.path.join(output_dir, f"Morphological_Stability_Index_{y1}_{y2}.tif")

        with rasterio.open(sdb_maps[y1]) as src_1, rasterio.open(sdb_maps[y2]) as src_2:
            z1 = src_1.read(1).astype(np.float32)
            
            z2 = np.zeros_like(z1)
            reproject(
                source=rasterio.band(src_2, 1),
                destination=z2,
                src_transform=src_2.transform,
                src_crs=src_2.crs,
                dst_transform=src_1.transform,
                dst_crs=src_1.crs,
                resampling=Resampling.nearest
            )

            res_x, res_y = abs(src_1.transform.a), abs(src_1.transform.e)
            if src_1.crs and src_1.crs.is_geographic:
                lat_center = (src_1.bounds.top + src_1.bounds.bottom) / 2.0
                m_per_deg_lat = 111132.92
                m_per_deg_lon = 111412.84 * np.cos(np.radians(lat_center))
                pixel_area_m2 = (res_x * m_per_deg_lon) * (res_y * m_per_deg_lat)
            else:
                pixel_area_m2 = res_x * res_y

            nodata_1 = src_1.nodata if src_1.nodata is not None else -9999
            nodata_2 = src_2.nodata if src_2.nodata is not None else -9999

            abs_z1 = np.abs(z1)
            abs_z2 = np.abs(z2)
            valid = (abs_z1 > 0.1) & (abs_z2 > 0.1) & (z1 != nodata_1) & (z2 != nodata_2) & (abs_z1 < 40) & (abs_z2 < 40)

            import geopandas as gpd
            from rasterio.features import rasterize
            
            def apply_mask(shp_path, current_valid_mask):
                if not shp_path or not os.path.exists(shp_path):
                    return current_valid_mask
                try:
                    gdf = gpd.read_file(shp_path)
                    if gdf.crs:
                        try:
                            gdf = gdf.to_crs(src_1.crs.to_wkt())
                        except Exception:
                            try:
                                gdf = gdf.to_crs(src_1.crs)
                            except Exception:
                                pass
                    if not gdf.empty:
                        geom = [g for g in gdf.geometry if g is not None]
                        shp_mask = rasterize(geom, out_shape=z1.shape, transform=src_1.transform, fill=0, default_value=1, dtype=np.uint8)
                        feedback.pushInfo(f"✂️ Applied Study Area OSW Polygon mask for Volume Analytics ({os.path.basename(shp_path)}).")
                        return current_valid_mask & (shp_mask == 1)
                except Exception as e:
                    feedback.pushWarning(f"⚠️ Could not apply OSW Polygon mask ({shp_path}): {e}")
                return current_valid_mask
                
            valid = apply_mask(osw_shp, valid)

            # Uncertainties
            if uncertainty_maps.get(y1) and os.path.exists(uncertainty_maps[y1]):
                with rasterio.open(uncertainty_maps[y1]) as u_src:
                    u1 = np.zeros_like(z1)
                    reproject(
                        source=rasterio.band(u_src, 1),
                        destination=u1,
                        src_transform=u_src.transform,
                        src_crs=u_src.crs,
                        dst_transform=src_1.transform,
                        dst_crs=src_1.crs,
                        resampling=Resampling.nearest
                    )
            else:
                u1 = np.full_like(z1, 0.4)

            if uncertainty_maps.get(y2) and os.path.exists(uncertainty_maps[y2]):
                with rasterio.open(uncertainty_maps[y2]) as u_src:
                    u2 = np.zeros_like(z1)
                    reproject(
                        source=rasterio.band(u_src, 1),
                        destination=u2,
                        src_transform=u_src.transform,
                        src_crs=u_src.crs,
                        dst_transform=src_1.transform,
                        dst_crs=src_1.crs,
                        resampling=Resampling.nearest
                    )
            else:
                u2 = np.full_like(z1, 0.4)

            # MSI and Robust Trend Prep
            z_stack = []
            years_to_stack = all_years if is_overall and all_years else [y1, y2]
            
            for yr in years_to_stack:
                if yr == y1:
                    z_stack.append(z1)
                elif yr == y2:
                    z_stack.append(z2)
                else:
                    with rasterio.open(sdb_maps[yr]) as yr_src:
                        z_yr = np.full_like(z1, -9999.0)
                        reproject(
                            source=rasterio.band(yr_src, 1),
                            destination=z_yr,
                            src_transform=yr_src.transform,
                            src_crs=yr_src.crs,
                            dst_transform=src_1.transform,
                            dst_crs=src_1.crs,
                            resampling=Resampling.nearest
                        )
                        z_stack.append(z_yr)

            z_arr = np.array(z_stack)
            z_arr[z_arr == -9999.0] = np.nan
            z_arr[z_arr == nodata_1] = np.nan
            z_arr[z_arr == nodata_2] = np.nan
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                z_std = np.nanstd(z_arr, axis=0)
                z_mean = np.nanmean(z_arr, axis=0)
            
            z_mean[np.isnan(z_mean) | (z_mean == 0)] = 0.001
            z_std[np.isnan(z_std)] = 0.0

            # Delta Z calculation (Robust Linear Regression if overall trend)
            delta_z = np.zeros_like(z1)
            sigma_delta = np.zeros_like(z1)
            
            if is_overall and all_years and len(all_years) > 2 and overall_trend_method == "Long-term Trend":
                # Linear Regression: delta_z = slope * total_years
                t = np.array(years_to_stack) - years_to_stack[0] # time in years
                t_mean = np.mean(t)
                t_diff = t - t_mean
                t_diff_2d = t_diff[:, None, None] # Broadcast to (N, H, W)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    z_diff = z_arr - z_mean
                    numerator = np.nansum(t_diff_2d * z_diff, axis=0)
                    denominator = np.sum(t_diff**2)
                    slope = numerator / denominator
                    
                    # Standard Error of the Regression Slope
                    y_hat = z_mean + slope * t_diff_2d
                    residuals = z_arr - y_hat
                    n = len(years_to_stack)
                    ss_res = np.nansum(residuals**2, axis=0)
                    # Add epsilon to denominator to avoid division by zero
                    se_slope = np.sqrt(ss_res / max(n - 2, 1)) / np.sqrt(denominator + 1e-6)
                
                slope[np.isnan(slope) | np.isinf(slope)] = 0.0
                se_slope[np.isnan(se_slope) | np.isinf(se_slope)] = 0.001
                
                total_years = years_to_stack[-1] - years_to_stack[0]
                delta_z[valid] = slope[valid] * total_years
                sigma_delta[valid] = se_slope[valid] * total_years
            else:
                # Sequential simple difference OR Net Difference for overall trend
                delta_z[valid] = z2[valid] - z1[valid]
                sigma_delta = np.sqrt(u1**2 + u2**2)

            sigma_delta[sigma_delta == 0] = 0.001

            z_score = np.zeros_like(z1)
            z_score[valid] = delta_z[valid] / sigma_delta[valid]

            statcd_class = np.full(z1.shape, 255, dtype=np.uint8)
            statcd_class[valid] = 1

            # Determine if map is Positive (Depth) or Negative (Elevation)
            is_positive_depth = np.nanmean(z1[valid]) > 0

            # If Depth is positive (e.g., 5m -> 15m means getting deeper)
            #   delta_z > 0 is Erosion
            #   delta_z < 0 is Accretion
            # If Depth is negative (e.g., -5m -> -15m means getting deeper)
            #   delta_z < 0 is Erosion
            #   delta_z > 0 is Accretion
            
            if is_positive_depth:
                true_accretion = valid & (z_score < -1.96)
                true_erosion = valid & (z_score > 1.96)
            else:
                true_accretion = valid & (z_score > 1.96)
                true_erosion = valid & (z_score < -1.96)

            # Apply Spatial Coherence Filter (Minimum Mapping Unit - MMU) to eliminate isolated single-pixel noise artifacts (Lane & Chandler, Wheaton et al. 2010)
            try:
                from scipy.ndimage import binary_opening
                struct_elem = np.ones((3, 3), dtype=bool)
                true_accretion = binary_opening(true_accretion, structure=struct_elem)
                true_erosion = binary_opening(true_erosion, structure=struct_elem)
            except Exception:
                pass

            statcd_class[true_accretion] = 2
            statcd_class[true_erosion] = 3

            # Compute volumes with explicit sign consistency
            if is_positive_depth:
                accretion_vol_m3 = float(np.sum(-delta_z[true_accretion]) * pixel_area_m2)
                erosion_vol_m3 = float(np.sum(delta_z[true_erosion]) * pixel_area_m2)
            else:
                accretion_vol_m3 = float(np.sum(delta_z[true_accretion]) * pixel_area_m2)
                erosion_vol_m3 = float(np.sum(-delta_z[true_erosion]) * pixel_area_m2)

            net_sediment_balance_m3 = accretion_vol_m3 - erosion_vol_m3

            profile = src_1.profile.copy()
            profile.update(count=1, dtype=rasterio.uint8, nodata=255, compress="lzw")
            with rasterio.open(statcd_raster_path, "w", **profile) as dst:
                dst.write(statcd_class, 1)

            # Uncertainty-Aware Morphological Stability Index (Isolating True Physical Variance from Measurement Uncertainty)
            # σ_obs² = σ_true² + σ_u²  =>  σ_true = sqrt(max(0, σ_obs² - σ_u²))
            sigma_u_mean_sq = (u1**2 + u2**2) / 2.0
            z_obs_var = z_std**2
            z_true_var = np.maximum(0.0, z_obs_var - sigma_u_mean_sq)
            z_true_std = np.sqrt(z_true_var)

            msi = np.full_like(z1, -9999.0, dtype=np.float32)
            # Add smoothing constant (0.5m) to denominator to prevent inflation in very shallow water
            msi[valid] = np.clip(1.0 - (z_true_std[valid] / (np.abs(z_mean[valid]) + 0.5)), 0.0, 1.0)

            profile_float = src_1.profile.copy()
            profile_float.update(count=1, dtype=rasterio.float32, nodata=-9999.0, compress="lzw")
            with rasterio.open(msi_raster_path, "w", **profile_float) as dst:
                dst.write(msi, 1)

        feedback.pushInfo(
            f"✅ StatCD Change Detection ({y1} -> {y2}):\n"
            f"   - Accretion: +{accretion_vol_m3:,.2f} m³ | Erosion: -{erosion_vol_m3:,.2f} m³ | Net: {net_sediment_balance_m3:+,.2f} m³"
        )

        return {
            "period": f"{y1}_{y2}",
            "first_year": y1,
            "last_year": y2,
            "statcd_raster_path": statcd_raster_path,
            "msi_raster_path": msi_raster_path,
            "accretion_volume_m3": accretion_vol_m3,
            "erosion_volume_m3": erosion_vol_m3,
            "net_sediment_balance_m3": net_sediment_balance_m3,
            "is_overall": is_overall,
            "analysis_type": "Overall Trend" if is_overall else analysis_type
        }

    def process_temporal_change(
        self,
        sdb_maps: Dict[int, str],
        uncertainty_maps: Dict[int, str],
        output_dir: str,
        feedback: QgsProcessingFeedback,
        osw_shp: str = None,
        overall_trend_method: str = "Long-term Trend",
        comparison_mode: str = "Sequential (Year-to-Year)",
    ) -> List[Dict[str, Any]]:
        """
        Executes multi-year bathymetric change detection sequentially or against baseline year, plus overall.
        """
        feedback.pushInfo(f"📈 [TEMPORAL ANALYTICS] Computing bathymetric evolution ({comparison_mode})...")

        years = sorted(list(sdb_maps.keys()))
        if len(years) < 2:
            feedback.pushWarning("At least 2 years required for temporal change analysis.")
            return []

        results = []
        
        # 1. Pairwise Analysis (Sequential vs Baseline Reference)
        if comparison_mode == "Baseline Reference (First Year Fixed)":
            y1 = years[0]
            for i in range(1, len(years)):
                y2 = years[i]
                res_pair = self._compute_pair(
                    y1, y2, sdb_maps, uncertainty_maps,
                    output_dir, feedback, osw_shp,
                    is_overall=False,
                    analysis_type="Baseline Reference"
                )
                results.append(res_pair)
        else: # Sequential (Year-to-Year)
            for i in range(len(years) - 1):
                y1, y2 = years[i], years[i+1]
                res_seq = self._compute_pair(
                    y1, y2, sdb_maps, uncertainty_maps,
                    output_dir, feedback, osw_shp,
                    is_overall=False,
                    analysis_type="Sequential"
                )
                results.append(res_seq)

        # 2. Overall Time Span (Robust Trend Analysis)
        first_yr, last_yr = years[0], years[-1]
        res_overall = self._compute_pair(
            first_yr, last_yr, sdb_maps, uncertainty_maps, 
            output_dir, feedback, osw_shp, 
            is_overall=True, all_years=years,
            overall_trend_method=overall_trend_method,
            analysis_type="Overall Trend"
        )
        results.append(res_overall)

        return results
