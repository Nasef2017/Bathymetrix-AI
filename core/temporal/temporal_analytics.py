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

    def _estimate_qr_sigma(self, sdb_raster_path, icesat_shp_path, depth_field, qr_alpha, feedback):
        """
        Fits QuantileRegressor at lower (alpha) and upper (1-alpha) quantiles
        on ICESat-2 training data, then predicts a spatially-adaptive sigma map.
        Returns a 2D np.ndarray matching the SDB raster shape.
        """
        if not sdb_raster_path or not os.path.exists(sdb_raster_path) or not icesat_shp_path or not os.path.exists(icesat_shp_path):
            return None

        try:
            from sklearn.linear_model import QuantileRegressor
            from qgis.core import QgsVectorLayer
            try:
                from ...core.ml.trainers import extract_samples
            except Exception:
                from Bathymetrix_AI.core.ml.trainers import extract_samples

            vec_layer = QgsVectorLayer(icesat_shp_path, "pts", "ogr")
            if not vec_layer or not vec_layer.isValid():
                return None

            # Extract samples (X: reflectance bands, y: depth)
            X, y, weights, coords = extract_samples(sdb_raster_path, vec_layer, depth_field, None, 0)
            
            if X is None or len(y) < 5:
                feedback.pushWarning(f"⚠️ Not enough training points for QR on {os.path.basename(sdb_raster_path)}. Falling back to Classical.")
                return None

            alpha_lo = (1.0 - qr_alpha) / 2.0
            alpha_hi = 1.0 - alpha_lo

            feedback.pushInfo(f"⏳ Fitting Quantile Regression Models (alpha={alpha_lo:.3f} and {alpha_hi:.3f}) for {os.path.basename(sdb_raster_path)}...")
            
            qr_lo = QuantileRegressor(quantile=alpha_lo, alpha=0)
            qr_hi = QuantileRegressor(quantile=alpha_hi, alpha=0)

            qr_lo.fit(X, y)
            qr_hi.fit(X, y)

            with rasterio.open(sdb_raster_path) as src:
                # Read all bands for prediction
                img_data = src.read()
                bands, height, width = img_data.shape
                # Reshape for prediction, ignoring nodata/nan
                nodata = src.nodata if src.nodata is not None else -9999
                
                # We need to flatten and filter valid pixels
                img_flat = img_data.reshape(bands, -1).T  # shape: (pixels, bands)
                
                # valid mask: all bands are not nodata and not nan
                valid_mask = np.all((img_flat != nodata) & ~np.isnan(img_flat), axis=1)
                
                if not np.any(valid_mask):
                    return None
                    
                X_pred = img_flat[valid_mask, :]
                
                # Predict
                pred_lo = qr_lo.predict(X_pred)
                pred_hi = qr_hi.predict(X_pred)
                
                from scipy import stats
                z_crit = stats.norm.ppf(1 - (1 - qr_alpha) / 2)
                
                # Calculate sigma: (q_hi - q_lo) / (2 * z_crit)
                sigma_pred = (pred_hi - pred_lo) / (2 * z_crit)
                
                sigma_map = np.zeros(height * width, dtype=np.float32)
                sigma_map[valid_mask] = sigma_pred
                sigma_map = sigma_map.reshape(height, width)
                
                return sigma_map

        except Exception as e:
            feedback.pushWarning(f"⚠️ Quantile Regression failed for {os.path.basename(sdb_raster_path)}: {str(e)}. Falling back to Classical.")
            return None

    def _compute_pair(self, y1, y2, sdb_maps, uncertainty_maps, output_dir, feedback, osw_shp, is_overall=False, all_years=None, overall_trend_method="Long-term Trend", analysis_type="Sequential", target_roi_path=None, uncertainty_mode="classical", qr_confidence=0.95, qr_sigma_1=None, qr_sigma_2=None):
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

            from rasterio.features import rasterize
            import json
            from qgis.core import QgsVectorLayer, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
            
            def apply_mask(shp_path, current_valid_mask):
                if not shp_path:
                    return current_valid_mask
                shp_path = str(shp_path).split('|')[0]
                if not os.path.exists(shp_path):
                    return current_valid_mask
                try:
                    vlayer = QgsVectorLayer(shp_path, "mask", "ogr")
                    if not vlayer.isValid():
                        return current_valid_mask
                    
                    target_crs = QgsCoordinateReferenceSystem(src_1.crs.to_wkt())
                    transform = None
                    if vlayer.crs() != target_crs:
                        transform = QgsCoordinateTransform(vlayer.crs(), target_crs, QgsProject.instance())
                    
                    geoms = []
                    for feat in vlayer.getFeatures():
                        geom = feat.geometry()
                        if geom.isNull(): continue
                        if transform:
                            try:
                                geom.transform(transform)
                            except Exception:
                                pass
                        geoms.append(json.loads(geom.asJson()))
                        
                    if not geoms:
                        return current_valid_mask
                        
                    shp_mask = rasterize(geoms, out_shape=z1.shape, transform=src_1.transform, fill=0, default_value=1, dtype=np.uint8)
                    feedback.pushInfo(f"✂️ Applied polygon mask ({os.path.basename(shp_path)}).")
                    return current_valid_mask & (shp_mask == 1)
                except Exception as e:
                    feedback.pushWarning(f"⚠️ Could not apply polygon mask ({shp_path}): {e}")
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
                
                if uncertainty_mode == "quantile_regression" and qr_sigma_1 is not None and qr_sigma_2 is not None:
                    sigma_delta[valid] = np.sqrt(qr_sigma_1[valid]**2 + qr_sigma_2[valid]**2)
                    sigma_delta[valid] = np.maximum(sigma_delta[valid], 0.05) # Floor
                else:
                    sigma_delta[valid] = np.sqrt(u1[valid]**2 + u2[valid]**2)

            sigma_delta[sigma_delta == 0] = 0.001

            from scipy import stats
            z_crit = stats.norm.ppf(1 - (1 - qr_confidence) / 2) if uncertainty_mode == "quantile_regression" else 1.96

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
                true_accretion = valid & (z_score < -z_crit)
                true_erosion = valid & (z_score > z_crit)
            else:
                true_accretion = valid & (z_score > z_crit)
                true_erosion = valid & (z_score < -z_crit)

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

            # Calculate Target ROI Volumes if provided
            target_accretion_vol_m3 = None
            target_erosion_vol_m3 = None
            target_net_balance_m3 = None

            if target_roi_path and os.path.exists(target_roi_path):
                target_valid = apply_mask(target_roi_path, valid)
                if is_positive_depth:
                    target_accretion_vol_m3 = float(np.sum(-delta_z[target_valid & (z_score < -z_crit)]) * pixel_area_m2)
                    target_erosion_vol_m3 = float(np.sum(delta_z[target_valid & (z_score > z_crit)]) * pixel_area_m2)
                else:
                    target_accretion_vol_m3 = float(np.sum(delta_z[target_valid & (z_score > z_crit)]) * pixel_area_m2)
                    target_erosion_vol_m3 = float(np.sum(-delta_z[target_valid & (z_score < -z_crit)]) * pixel_area_m2)
                target_net_balance_m3 = target_accretion_vol_m3 - target_erosion_vol_m3

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
        if target_accretion_vol_m3 is not None:
            feedback.pushInfo(
                f"🎯 Target ROI Change Detection ({y1} -> {y2}):\n"
                f"   - Target Accretion: +{target_accretion_vol_m3:,.2f} m³ | Target Erosion: -{target_erosion_vol_m3:,.2f} m³ | Target Net: {target_net_balance_m3:+,.2f} m³"
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
            "target_accretion_volume_m3": target_accretion_vol_m3,
            "target_erosion_volume_m3": target_erosion_vol_m3,
            "target_net_sediment_balance_m3": target_net_balance_m3,
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
        target_roi_path: str = None,
        uncertainty_mode: str = "classical",
        qr_confidence: float = 0.95,
        training_maps: Dict[int, str] = None,
        depth_field: str = None
    ) -> List[Dict[str, Any]]:
        """
        Executes multi-year bathymetric change detection sequentially or against baseline year, plus overall.
        """
        feedback.pushInfo(f"📈 [TEMPORAL ANALYTICS] Computing bathymetric evolution ({comparison_mode})...")

        years = sorted(list(sdb_maps.keys()))
        if len(years) < 2:
            feedback.pushWarning("At least 2 years required for temporal change analysis.")
            return []

        qr_sigmas = {}
        if uncertainty_mode == "quantile_regression" and training_maps and depth_field:
            for yr in years:
                sdb_path = sdb_maps.get(yr)
                shp_path = training_maps.get(yr)
                if sdb_path and shp_path:
                    sigma_map = self._estimate_qr_sigma(sdb_path, shp_path, depth_field, qr_confidence, feedback)
                    if sigma_map is not None:
                        qr_sigmas[yr] = sigma_map
                        # Save QR uncertainty map
                        try:
                            qr_out_path = os.path.join(output_dir, f"Quantile_Uncertainty_{yr}.tif")
                            with rasterio.open(sdb_path) as src:
                                profile = src.profile.copy()
                                profile.update(count=1, dtype=rasterio.float32, nodata=-9999.0, compress="lzw")
                                with rasterio.open(qr_out_path, "w", **profile) as dst:
                                    dst.write(sigma_map.astype(np.float32), 1)
                            feedback.pushInfo(f"💾 Saved QR Uncertainty Map: {qr_out_path}")
                        except Exception as e:
                            feedback.pushWarning(f"⚠️ Failed to save QR Uncertainty map for {yr}: {e}")
                    else:
                        feedback.pushWarning(f"⚠️ QR Sigma map generation failed for {yr}. Will fallback to classical.")

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
                    analysis_type="Baseline Reference",
                    target_roi_path=target_roi_path,
                    uncertainty_mode=uncertainty_mode,
                    qr_confidence=qr_confidence,
                    qr_sigma_1=qr_sigmas.get(y1),
                    qr_sigma_2=qr_sigmas.get(y2)
                )
                results.append(res_pair)
        else: # Sequential (Year-to-Year)
            for i in range(len(years) - 1):
                y1, y2 = years[i], years[i+1]
                res_seq = self._compute_pair(
                    y1, y2, sdb_maps, uncertainty_maps,
                    output_dir, feedback, osw_shp,
                    is_overall=False,
                    analysis_type="Sequential",
                    target_roi_path=target_roi_path,
                    uncertainty_mode=uncertainty_mode,
                    qr_confidence=qr_confidence,
                    qr_sigma_1=qr_sigmas.get(y1),
                    qr_sigma_2=qr_sigmas.get(y2)
                )
                results.append(res_seq)

        # 2. Overall Time Span (Robust Trend Analysis)
        first_yr, last_yr = years[0], years[-1]
        res_overall = self._compute_pair(
            first_yr, last_yr, sdb_maps, uncertainty_maps, 
            output_dir, feedback, osw_shp, 
            is_overall=True, all_years=years,
            overall_trend_method=overall_trend_method,
            analysis_type="Overall Trend",
            target_roi_path=target_roi_path,
            uncertainty_mode=uncertainty_mode,
            qr_confidence=qr_confidence,
            qr_sigma_1=qr_sigmas.get(first_yr),
            qr_sigma_2=qr_sigmas.get(last_yr)
        )
        results.append(res_overall)

        return results
