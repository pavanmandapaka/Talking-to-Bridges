import os
from pathlib import Path
import pandas as pd
import numpy as np

# Resolve the project root (the folder containing data/processed/sensors)
# so the script works no matter where it is run from.
_here = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in [_here.parent, *_here.parents]
                     if (p / "data/processed/sensors").exists()), Path.cwd())
SENSOR_DIR = PROJECT_ROOT / "data/processed/sensors"

def generate_eda_summary():
    manifest_path = str(SENSOR_DIR / "manifest.csv")
    if not os.path.exists(manifest_path):
        print("[ERROR] manifest.csv not found.")
        return

    manifest = pd.read_csv(manifest_path)
    if "output_file" in manifest.columns:  # drop rows the pipeline skipped (no clean file)
        skipped = int(manifest["output_file"].isna().sum())
        manifest = manifest[manifest["output_file"].notna()].reset_index(drop=True)
        if skipped:
            print(f"[INFO] Ignoring {skipped} skipped manifest row(s) (no clean file).")
    print("=== DATASET MANIFEST OVERVIEW ===")
    print(f"Total recordings in manifest: {len(manifest)}")
    if 'condition' in manifest.columns:
        print(manifest['condition'].value_counts())
    
    sensor_cols = [f"Sensor_{i}" for i in range(1, 6)]
    summary_rows = []

    for _, row in manifest.iterrows():
        out_rel_path = row.get('output_file')
        if not out_rel_path or pd.isna(out_rel_path):
            continue
        
        full_path = os.path.join(str(SENSOR_DIR), str(out_rel_path))
        if not os.path.exists(full_path):  # file may sit in damaged/ or undamaged/
            hit = next(SENSOR_DIR.rglob(Path(str(out_rel_path)).name), None)
            full_path = str(hit) if hit else full_path
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)

            damage_val = df['Damage_Level'].iloc[0] if 'Damage_Level' in df.columns else 'N/A'
            specimen_val = df['Specimen'].iloc[0] if 'Specimen' in df.columns else 'N/A'
            test_type_val = df['Test_Type'].iloc[0] if 'Test_Type' in df.columns else 'N/A'
            hit_group_val = df['Hit_Group'].iloc[0] if 'Hit_Group' in df.columns else 'N/A'

            stats = {
                'source_file': row.get('source_file'),
                'condition': row.get('condition', 'Unknown'),
                'damage_level': damage_val,
                'specimen': specimen_val,
                'test_type': test_type_val,
                'hit_group': hit_group_val,
                'row_count': len(df)
            }
            
            for col in sensor_cols:
                if col in df.columns:
                    stats[f'{col}_mean'] = df[col].mean()
                    stats[f'{col}_std'] = df[col].std()
                    stats[f'{col}_p2p'] = df[col].max() - df[col].min()
            
            summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    output_csv = str(SENSOR_DIR / "eda_summary_metrics.csv")
    summary_df.to_csv(output_csv, index=False)
    
    print("\n=== EDA METRICS GENERATED ===")
    print(f"Summary metrics exported to: {output_csv}")
    print(summary_df.head())

if __name__ == "__main__":
    generate_eda_summary()