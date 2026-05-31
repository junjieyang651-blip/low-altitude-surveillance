"""
可视化脚本：生成离线 HTML 态势地图

不依赖 folium 等第三方库，直接生成包含 Leaflet.js 的原生 HTML，
在任何浏览器中打开即可查看航迹、冲突、异常事件。

使用方法:
    python visualize.py --output ./output/situation_map.html
（需先运行 main.py 生成 output 目录下的中间结果）
"""

import os
import sys
import json
import argparse

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.coords import enu_to_wgs84


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_map_html(datasets_dir: str, output_dir: str, out_html: str):
    # 读取原始数据（取少量样本避免HTML过大）
    tracks_geojson = {"type": "FeatureCollection", "features": []}
    points_geojson = {"type": "FeatureCollection", "features": []}

    # 颜色配置
    source_colors = {
        "adsb": "#e74c3c",      # 红
        "remote_id": "#3498db", # 蓝
        "radar": "#2ecc71",     # 绿
        "spectrum": "#9b59b6",  # 紫
        "fused": "#f39c12",     # 橙
    }

    # ENU 参考原点（从 main.py 输出中读取，或重新计算）
    ref_lat, ref_lon, ref_alt = 30.451439, 114.009479, 0.0

    def add_track(df, src_name, tid):
        color = source_colors.get(src_name, "#7f8c8d")
        coords = []
        for _, r in df.iterrows():
            if not pd.isna(r.get("lat")) and not pd.isna(r.get("lon")):
                coords.append([float(r["lat"]), float(r["lon"])])
        if len(coords) < 2:
            return
        feature = {
            "type": "Feature",
            "properties": {
                "source": src_name,
                "target_id": str(tid),
                "color": color,
                "weight": 3 if src_name == "fused" else 2,
                "opacity": 0.8,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[c[1], c[0]] for c in coords],  # GeoJSON: [lon, lat]
            },
        }
        tracks_geojson["features"].append(feature)

        # 起点和终点标记
        for idx, c in enumerate([coords[0], coords[-1]]):
            pt = {
                "type": "Feature",
                "properties": {
                    "source": src_name,
                    "target_id": str(tid),
                    "point_type": "start" if idx == 0 else "end",
                    "color": color,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [c[1], c[0]],
                },
            }
            points_geojson["features"].append(pt)

    # 加载原始数据并采样
    data_dir = datasets_dir
    for src in ["adsb", "remote_id", "radar"]:
        for ext in [".xlsx", ".csv"]:
            path = os.path.join(data_dir, f"ad_{src}{ext}")
            if os.path.exists(path):
                df = pd.read_excel(path) if ext == ".xlsx" else pd.read_csv(path)
                lat_col = "lat" if "lat" in df.columns else "latitude"
                lon_col = "lon" if "lon" in df.columns else "longitude"
                tid_col = "icao" if src == "adsb" else ("uas_id" if src == "remote_id" else "track_number")
                if tid_col not in df.columns:
                    continue
                # 限制每源最多画50条航迹，每条最多100点，防止HTML爆炸
                tids = df[tid_col].unique()[:50]
                for tid in tids:
                    sub = df[df[tid_col] == tid].head(100)
                    sub = sub.rename(columns={lat_col: "lat", lon_col: "lon"})
                    add_track(sub, src, tid)
                break

    # 加载融合航迹（新格式：fused_tracks_all.csv）
    fused_path = os.path.join(output_dir, "fused_tracks_all.csv")
    if os.path.exists(fused_path):
        fdf = pd.read_csv(fused_path)
        if "e" in fdf.columns and "n" in fdf.columns:
            # 反算 lat/lon
            lats, lons, _ = enu_to_wgs84(
                fdf["e"].values, fdf["n"].values, fdf.get("u", 0).values,
                ref_lat, ref_lon, ref_alt
            )
            fdf["lat"] = lats
            fdf["lon"] = lons
        # 限制最多画30条融合航迹
        stids = fdf["system_track_id"].unique()[:30]
        for stid in stids:
            sub = fdf[fdf["system_track_id"] == stid].dropna(subset=["lat", "lon"]).head(100)
            add_track(sub, "fused", stid)

    # 加载异常事件
    anomaly_markers = []
    anomaly_path = os.path.join(output_dir, "anomaly_events.csv")
    if os.path.exists(anomaly_path):
        adf = pd.read_csv(anomaly_path)
        # 异常事件缺少 lat/lon，这里无法直接标在地图上
        # 仅作为统计信息展示在面板中
        anomaly_summary = adf.groupby("type").size().to_dict()
    else:
        anomaly_summary = {}

    # 加载冲突事件
    conflict_markers = []
    conflict_path = os.path.join(output_dir, "conflict_events.csv")
    if os.path.exists(conflict_path):
        cdf = pd.read_csv(conflict_path)
        conflict_summary = cdf.groupby("level").size().to_dict()
    else:
        conflict_summary = {}

    # 生成 HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>低空监视态势系统</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin: 0; font-family: "Microsoft YaHei", sans-serif; }}
  #map {{ height: 100vh; width: 100%; }}
  .info-panel {{
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2); min-width: 220px; max-width: 320px;
    font-size: 13px;
  }}
  .info-panel h3 {{ margin: 0 0 10px 0; font-size: 15px; color: #2c3e50; }}
  .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
  .legend-color {{ width: 16px; height: 3px; margin-right: 8px; }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 4px; }}
  .badge-critical {{ background: #e74c3c; color: white; }}
  .badge-warning {{ background: #f39c12; color: white; }}
  .badge-caution {{ background: #f1c40f; color: #333; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel">
  <h3>低空监视态势面板</h3>
  <div><b>图例</b></div>
  <div class="legend-item"><div class="legend-color" style="background:{source_colors['adsb']}"></div>ADS-B</div>
  <div class="legend-item"><div class="legend-color" style="background:{source_colors['remote_id']}"></div>Remote ID</div>
  <div class="legend-item"><div class="legend-color" style="background:{source_colors['radar']}"></div>雷达</div>
  <div class="legend-item"><div class="legend-color" style="background:{source_colors['fused']}"></div>融合航迹</div>
  <hr style="border:0;border-top:1px solid #ddd;margin:10px 0;">
  <div><b>冲突预警</b></div>
  {"<div style='color:#7f8c8d'>无冲突数据</div>" if not conflict_summary else ""}
'''
    for lvl, cnt in sorted(conflict_summary.items(), key=lambda x: {"critical":0,"warning":1,"caution":2}.get(x[0],3)):
        cls = f"badge-{lvl}"
        html += f'  <div>{lvl.upper()}: <span class="badge {cls}">{cnt}</span></div>\n'

    html += '''  <hr style="border:0;border-top:1px solid #ddd;margin:10px 0;">
  <div><b>异常事件</b></div>
'''
    if not anomaly_summary:
        html += '  <div style="color:#7f8c8d">无异常数据</div>\n'
    for typ, cnt in sorted(anomaly_summary.items(), key=lambda x: -x[1])[:8]:
        html += f'  <div>{typ}: {cnt}</div>\n'

    html += f'''</div>
<script>
  var map = L.map('map').setView([30.451, 114.009], 12);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 18,
    attribution: 'OpenStreetMap'
  }}).addTo(map);

  var tracksData = {json.dumps(tracks_geojson, ensure_ascii=False)};
  var pointsData = {json.dumps(points_geojson, ensure_ascii=False)};

  // 绘制航迹线
  L.geoJSON(tracksData, {{
    style: function(feature) {{
      return {{
        color: feature.properties.color,
        weight: feature.properties.weight,
        opacity: feature.properties.opacity
      }};
    }},
    onEachFeature: function(feature, layer) {{
      layer.bindPopup(feature.properties.source.toUpperCase() + ' 航迹: ' + feature.properties.target_id);
    }}
  }}).addTo(map);

  // 绘制起止点
  L.geoJSON(pointsData, {{
    pointToLayer: function(feature, latlng) {{
      return L.circleMarker(latlng, {{
        radius: 4,
        fillColor: feature.properties.color,
        color: '#fff',
        weight: 1,
        opacity: 1,
        fillOpacity: 0.9
      }});
    }},
    onEachFeature: function(feature, layer) {{
      layer.bindPopup(feature.properties.source.toUpperCase() + ' ' + feature.properties.point_type + ': ' + feature.properties.target_id);
    }}
  }}).addTo(map);

  // 自适应视野
  var allLayers = [];
  map.eachLayer(function(layer){{
    if(layer.getBounds) allLayers.push(layer);
  }});
  if(allLayers.length > 0){{
    var group = new L.featureGroup(allLayers);
    map.fitBounds(group.getBounds().pad(0.1));
  }}
</script>
</body>
</html>
'''

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"态势地图已生成: {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./shuju", help="原始数据目录")
    parser.add_argument("--output", default="./output", help="main.py 输出目录")
    parser.add_argument("--html", default="./output/situation_map.html", help="输出HTML路径")
    args = parser.parse_args()
    generate_map_html(args.data, args.output, args.html)
