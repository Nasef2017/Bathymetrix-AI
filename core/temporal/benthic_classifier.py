import os
import numpy as np
import rasterio
from typing import Dict, Any, Tuple
try:
    from qgis.core import QgsProcessingFeedback
except ImportError:
    QgsProcessingFeedback = object


class BenthicVegetationClassifier:
    """
    Generic Benthic Vegetation & Substrate Classifier using
    Depth-Invariant Substrate Indices (DII - Lyzenga Method).
    
    Removes water attenuation using predicted SDB depth Z_t(x,y)
    to classify marine vegetation (seagrass/macroalgae), sand, rock, and deep water.
    """

    def __init__(self, coastal_idx: int = 1, blue_idx: int = 2, green_idx: int = 3, red_idx: int = 4):
        self.coastal_idx = coastal_idx
        self.blue_idx = blue_idx
        self.green_idx = green_idx
        self.red_idx = red_idx

    def process_year_benthic(
        self,
        year: int,
        image_path: str,
        sdb_depth_path: str,
        output_dir: str,
        feedback: QgsProcessingFeedback,
    ) -> Dict[str, Any]:
        """
        Calculates Depth-Invariant Indices (DII) and classifies benthic substrate for a year.
        """
        feedback.pushInfo(f"🌿 [BENTHIC CLASSIFIER] Processing year {year} benthic habitat mapping...")

        benthic_map_path = os.path.join(output_dir, f"Benthic_Habitat_{year}.tif")

        with rasterio.open(image_path) as img_src, rasterio.open(sdb_depth_path) as sdb_src:
            img_data = img_src.read()
            sdb_depth = sdb_src.read(1)
            nodata = img_src.nodata if img_src.nodata is not None else 0

            # Get bands (1-indexed input)
            blue = img_data[self.blue_idx - 1].astype(np.float32)
            green = img_data[self.green_idx - 1].astype(np.float32)
            red = img_data[self.red_idx - 1].astype(np.float32)

            # Use absolute depth to handle both positive and negative SDB conventions
            abs_depth = np.abs(sdb_depth)
            
            # Valid water mask (depth > 0 and reflectance > 0)
            valid_mask = (abs_depth > 0.1) & (blue > 0) & (green > 0) & (abs_depth < 35.0)

            # Compute deep water reflectance R_infinity (lowest 1% percentile of deep pixels)
            deep_mask = (abs_depth > 15.0) & (blue > 0)
            if np.sum(deep_mask) > 50:
                r_inf_b = np.percentile(blue[deep_mask], 5)
                r_inf_g = np.percentile(green[deep_mask], 5)
            else:
                r_inf_b = max(np.percentile(blue[valid_mask], 1), 0.001) if np.sum(valid_mask) > 0 else 0.001
                r_inf_g = max(np.percentile(green[valid_mask], 1), 0.001) if np.sum(valid_mask) > 0 else 0.001

            # Log-transform band signals: X_i = ln(R_i - R_inf)
            b_sub = np.maximum(blue - r_inf_b, 0.0001)
            g_sub = np.maximum(green - r_inf_g, 0.0001)

            x_b = np.log(b_sub)
            x_g = np.log(g_sub)

            # Estimate attenuation ratio ki/kj from depth relationship
            # ki/kj approx ratio of log slope over depth
            ki_kj = 0.75 # Default optical attenuation ratio blue/green in clear coastal water

            # Compute Lyzenga Depth-Invariant Index: DII_bg = X_b - (ki/kj)*X_g
            dii_bg = np.zeros_like(blue)
            dii_bg[valid_mask] = x_b[valid_mask] - ki_kj * x_g[valid_mask]

            # Reconstruct bottom reflectance R0 (water-column corrected)
            # R0 = (R - R_inf) * exp(2 * K * depth)
            k_green = 0.12 # Mean green diffuse attenuation coefficient
            r0_green = np.zeros_like(green)
            r0_green[valid_mask] = g_sub[valid_mask] * np.exp(2.0 * k_green * abs_depth[valid_mask])

            # Classify Substrate:
            # 1: Vegetation
            # 2: Sandy Substrate / Soft Sediment
            # 3: Hard Bottom / Rock / Coral
            import glob
            import json
            from rasterio.features import rasterize
            from qgis.core import QgsVectorLayer
            from sklearn.ensemble import RandomForestClassifier

            class_map = np.full(sdb_depth.shape, 255, dtype=np.uint8)
            
            year_dir = os.path.dirname(image_path)
            shp_files = glob.glob(os.path.join(year_dir, "*benthic*.shp")) + glob.glob(os.path.join(year_dir, "*train*.shp"))
            shp_files = list(set(shp_files))
            
            is_supervised = False

            if len(shp_files) > 0 and np.sum(valid_mask) > 0:
                shp_path = shp_files[0]
                feedback.pushInfo(f"      [ML] Found training data: {os.path.basename(shp_path)}. Switching to Supervised Classification.")
                layer = QgsVectorLayer(shp_path, "training", "ogr")
                if layer.isValid():
                    fields = [f.name().lower() for f in layer.fields()]
                    id_field = None
                    for f in ["class_id", "id", "class", "label"]:
                        if f in fields:
                            id_field = layer.fields()[fields.index(f)].name()
                            break
                    if not id_field:
                        id_field = layer.fields()[0].name()
                    
                    shapes = []
                    for feat in layer.getFeatures():
                        geom = feat.geometry()
                        if not geom.isNull():
                            try:
                                geom_json = json.loads(geom.asJson())
                                val = int(feat[id_field])
                                shapes.append((geom_json, val))
                            except Exception:
                                pass
                    
                    if shapes:
                        train_raster = rasterize(
                            shapes,
                            out_shape=blue.shape,
                            transform=img_src.transform,
                            fill=0,
                            dtype=np.uint8
                        )
                        
                        train_mask = (train_raster > 0) & valid_mask
                        if np.sum(train_mask) > 10:
                            X_img = np.stack([
                                blue[valid_mask],
                                green[valid_mask],
                                red[valid_mask],
                                dii_bg[valid_mask],
                                r0_green[valid_mask],
                                abs_depth[valid_mask]
                            ], axis=-1)
                            
                            X_train = np.stack([
                                blue[train_mask],
                                green[train_mask],
                                red[train_mask],
                                dii_bg[train_mask],
                                r0_green[train_mask],
                                abs_depth[train_mask]
                            ], axis=-1)
                            
                            y_train = train_raster[train_mask]
                            
                            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_img)
                            
                            class_map[valid_mask] = y_pred
                            is_supervised = True
                            feedback.pushInfo(f"      [ML] Random Forest trained on {len(y_train)} pixels. Image classified.")

            if not is_supervised and np.sum(valid_mask) > 0:
                feedback.pushInfo("      [Rules] No training data found. Extracting Vegetation Only using Depth & DII.")
                dii_vals = dii_bg[valid_mask]

                # Differentiate between vegetation (green) and deep water darkness using depth
                # Vegetation is typically found where DII is low AND depth is reasonable (< 20m)
                max_veg_depth = 20.0
                veg = valid_mask & (dii_bg < np.median(dii_vals)) & (abs_depth < max_veg_depth)

                class_map[veg] = 1

            # ENFORCE VEGETATION ONLY: Remove any Sand (2) or Rock (3) from supervised predictions
            class_map[(class_map == 2) | (class_map == 3)] = 255

            # Write raster output
            profile = img_src.profile.copy()
            profile.update(count=1, dtype=rasterio.uint8, nodata=255, compress="lzw")

            with rasterio.open(benthic_map_path, "w", **profile) as dst:
                dst.write(class_map, 1)

            # Compute area statistics
            res_x, res_y = abs(img_src.transform.a), abs(img_src.transform.e)
            pixel_area_km2 = (res_x * res_y) / 1e6

            veg_area = float(np.sum(class_map == 1) * pixel_area_km2)
            sand_area = 0.0
            rock_area = 0.0

        feedback.pushInfo(f"✅ Benthic Classification Year {year}: Veg={veg_area:.2f}km²")

        return {
            "year": year,
            "benthic_map_path": benthic_map_path,
            "veg_area_km2": veg_area,
            "sand_area_km2": sand_area,
            "rock_area_km2": rock_area,
        }
