"""Versioned artifacts and the confidence-aware Digest Evidence Workspace."""

from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .stereo_contracts import ProcessedStereoResult, ProcessingDisposition


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _colorize(
    values: np.ndarray,
    validity: np.ndarray,
    colormap: int,
    reverse: bool = False,
) -> np.ndarray:
    finite = validity & np.isfinite(values)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if np.any(finite):
        low, high = np.percentile(values[finite], [2, 98])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((values - low) / (high - low), 0, 1)
        if reverse:
            scaled = 1.0 - scaled
        normalized[finite] = np.rint(scaled[finite] * 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, colormap)
    color[~finite] = (18, 23, 32)
    return color


def _write_digest(
    output_dir: Path,
    result: ProcessedStereoResult,
    confidence_u8: np.ndarray,
    *,
    has_source_images: bool,
) -> None:
    validity_u8 = result.disparity.validity.astype(np.uint8)
    depth_validity_u8 = result.depth.validity.astype(np.uint8)
    display_payload = {
        "width": int(confidence_u8.shape[1]),
        "height": int(confidence_u8.shape[0]),
        "confidence_base64": base64.b64encode(confidence_u8.tobytes()).decode(),
        "validity_base64": base64.b64encode(validity_u8.tobytes()).decode(),
        "depth_validity_base64": base64.b64encode(
            depth_validity_u8.tobytes()
        ).decode(),
        "depth_m_base64": base64.b64encode(
            np.asarray(result.depth.meters, dtype="<f4").tobytes()
        ).decode(),
    }
    (output_dir / "display_data.js").write_text(
        "window.STEREO_DISPLAY="
        + json.dumps(display_payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    diagnostic = result.disposition is ProcessingDisposition.DIAGNOSTIC
    status_title = (
        "Metric depth is diagnostic, not trusted"
        if diagnostic
        else "Metric depth is trusted"
    )
    reason = ", ".join(result.reason_codes) if result.reason_codes else "none"
    action = result.calibration.recommended_action or "No action required."
    hard_coverage = float(np.mean(result.disparity.validity) * 100)
    valid_depth = result.depth.meters[result.depth.validity]
    if valid_depth.size:
        depth_near, depth_far = np.percentile(valid_depth, [2, 98])
    else:
        depth_near, depth_far = np.nan, np.nan
    html_text = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>iPhone Stereo Depth Digest</title>
<style>
:root{color-scheme:dark;--bg:#080d18;--panel:#101827;--line:#26344a;--text:#eef4ff;--muted:#91a0b8;--cyan:#25c2d8;--amber:#f1b84b;--rose:#ef7188;--green:#42d39b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,sans-serif}header{height:68px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:#0c1320;border-bottom:1px solid var(--line)}h1,h2,p{margin-top:0}.brand{display:flex;gap:11px;align-items:center}.mark{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--cyan),#6c63ff);display:grid;place-items:center;font-weight:800}.chip{border:1px solid #725421;background:#302611;color:#ffd889;border-radius:999px;padding:6px 10px;font-size:12px}main{padding:20px 22px 50px}.limitation{border:1px solid #765922;background:linear-gradient(90deg,#2c2312,#1b1920);border-radius:10px;padding:14px 16px;display:flex;gap:14px}.limitation.trusted{border-color:#23684c;background:#10281f}.limitation strong{color:#ffd889}.limitation.trusted strong{color:#8ff0c4}.limitation p{margin:3px 0;color:#dbcda9}.source-section{margin-top:14px}.source-section h2{margin-bottom:9px}.source-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.source-frame{position:relative;overflow:hidden;border-radius:9px;border:1px solid #334155;background:#050914}.source-frame img{display:block;width:100%;height:auto}.depth-tools{display:grid;grid-template-columns:minmax(220px,1fr) minmax(220px,2fr);gap:16px;align-items:center;margin-bottom:12px}.depth-readout{padding:11px 13px;background:#07101d;border:1px solid #36516f;border-radius:8px}.depth-readout b{display:block;color:#fff;font-size:24px}.depth-inspector{position:relative;overflow:hidden;border-radius:9px;border:1px solid #334155;background:#050914;cursor:crosshair;touch-action:none}.depth-inspector img{display:block;width:100%;height:auto}.depth-inspector .depth-layer,.depth-inspector canvas{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}.depth-marker{display:none;position:absolute;width:17px;height:17px;border:2px solid white;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 2px #07101d;pointer-events:none;z-index:4}.depth-scale{height:11px;border-radius:9px;background:linear-gradient(90deg,#f0f921,#cc4778,#6a00a8,#0d0887);margin-top:9px}.grid{display:grid;grid-template-columns:230px minmax(0,1fr) 250px;gap:14px;margin-top:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px}.side{display:flex;flex-direction:column;gap:12px}.control label{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}.value{color:var(--cyan);font-size:21px;font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:var(--cyan)}.ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}.inspection{display:inline-block;border:1px solid #275365;background:#102b35;color:#9cebf5;padding:5px 8px;border-radius:6px;font-size:11px}.reset{border:0;background:none;color:var(--cyan);cursor:pointer}.muted{color:var(--muted)}.metric{padding:11px;background:#0b1220;border:1px solid #263247;border-radius:8px}.metric span{display:block;color:var(--muted);font-size:11px}.metric b{font-size:19px}.maps{display:grid;grid-template-columns:1fr 1fr;gap:12px}.map{position:relative;overflow:hidden;border-radius:9px;border:1px solid #334155;background:#050914;min-height:300px}.map img{display:block;width:100%;height:100%;object-fit:cover;position:absolute;inset:0}.map canvas{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}.label{position:absolute;left:10px;top:10px;z-index:2;background:#07101ddd;border:1px solid #42536c;border-radius:6px;padding:5px 8px}.label small{display:block;color:var(--muted)}.legend{height:10px;border-radius:9px;background:linear-gradient(90deg,#3b1c69,#2563eb,#2dd4bf,#facc15);margin:8px 0}.keys{font-size:11px;color:var(--muted)}code{overflow-wrap:anywhere}@media(max-width:900px){.grid{display:flex;flex-direction:column}.maps,.source-grid,.depth-tools{grid-template-columns:1fr}.map{min-height:360px}}
</style></head><body>
<header><div class="brand"><div class="mark">D</div><div><b>Depth Digest</b><small class="muted" style="display:block">Stereo Result Manifest v1</small></div></div><span class="chip">__DISPOSITION__</span></header>
<main><section class="limitation __STATUS_CLASS__"><div>__ICON__</div><div><strong>__STATUS_TITLE__</strong><p>Reason: <code>__REASON__</code></p><p>__ACTION__</p></div></section>
__SOURCE_SECTION__
<section class="card source-section"><h2>Rectified stereo pair</h2><p class="muted">These views share the same coordinates as disparity and metric depth. Compare the Disparity Map against the Ultra-Wide view on the left.</p><div class="source-grid"><div class="source-frame"><img src="rectified_ultrawide.png" alt="Rectified Ultra-Wide camera view"><div class="label"><b>Ultra-Wide · physical left</b><small>rectified reference for disparity</small></div></div><div class="source-frame"><img src="rectified_main.png" alt="Rectified Main camera view"><div class="label"><b>Main · physical right</b><small>rectified correspondence view</small></div></div></div></section>
<section class="card source-section"><h2>Interactive depth overlay</h2><p class="muted">Point at or tap an object to read its diagnostic metric depth. The image is the rectified Ultra-Wide view because it shares the Depth Map coordinates.</p><div class="depth-tools"><div class="control"><label><b>Depth overlay opacity</b><span class="value" id="overlayOpacityValue">55%</span></label><input id="overlayOpacity" type="range" min="0" max="1" step="0.01" value="0.55" aria-label="Depth overlay opacity"><div class="ticks"><span>Actual image</span><span>Depth colors</span></div></div><div class="depth-readout"><span class="muted">Object under pointer</span><b id="depthReadout">Point at or tap an object</b><span class="muted" id="depthEvidence">Depth and confidence will appear here.</span></div></div><div class="depth-inspector" id="depthInspector"><img src="rectified_ultrawide.png" alt="Rectified actual Ultra-Wide image"><img class="depth-layer" id="depthColorLayer" src="depth_map.png" alt="Metric depth color overlay"><canvas data-mask="depth-inspector"></canvas><div class="depth-marker" id="depthMarker"></div></div><div class="depth-scale"></div><div class="ticks"><span>Near · __DEPTH_NEAR__ m</span><span>Far · __DEPTH_FAR__ m</span></div></section>
<div class="grid"><aside class="side"><div class="card control"><label><b>Confidence threshold</b><span class="value" id="thresholdValue">0.00</span></label><input id="threshold" type="range" min="0" max="1" step="0.01" value="0" aria-label="Confidence threshold"><div class="ticks"><span>0 · all hard-valid</span><span>1 · strongest</span></div><div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px"><span class="inspection">Inspection only</span><button class="reset" id="reset">Reset</button></div><p class="muted" style="margin:11px 0 0">Normalized evidence, not a probability. This control never recomputes disparity or depth.</p></div><div class="metric"><span>Retained hard-valid pixels</span><b id="retainedHard">100.0%</b></div><div class="metric"><span>Displayed full-frame coverage</span><b id="retainedFrame">__HARD_COVERAGE__%</b></div><div class="card"><b>Confidence legend</b><div class="legend"></div><div class="keys">0 weak evidence · 1 strong evidence<br>Hatched = hard invalid<br>Dark = below threshold</div></div></aside>
<section class="maps"><div class="map"><img src="disparity_map.png" alt="Stored disparity preview"><canvas data-mask="disparity"></canvas><div class="label"><b>Disparity</b><small>physical Ultra-Wide left</small></div></div><div class="map"><img src="depth_map.png" alt="Stored metric depth preview"><canvas data-mask="depth"></canvas><div class="label"><b>Metric depth</b><small>same Displayed Pixel Set</small></div></div></section>
<aside class="side"><div class="card"><h2>Result evidence</h2><div class="metric"><span>Hard-valid frame coverage</span><b>__HARD_COVERAGE__%</b></div><div class="metric" style="margin-top:8px"><span>Trusted Depth Eligibility</span><b style="color:__TRUST_COLOR__">__TRUSTED__</b></div><div class="metric" style="margin-top:8px"><span>Matcher</span><b style="font-size:14px">Sliding Window 31×31</b></div></div><div class="card"><h2>Stored computation</h2><p class="muted">Disparity, confidence, hard validity, and metric depth were computed once by the shared processor. The slider intersects stored arrays only.</p><a href="stereo_result.json" style="color:var(--cyan)">Open manifest</a></div></aside></div></main>
<script src="display_data.js"></script><script>
const data=window.STEREO_DISPLAY;const confidence=Uint8Array.from(atob(data.confidence_base64),c=>c.charCodeAt(0));const validity=Uint8Array.from(atob(data.validity_base64),c=>c.charCodeAt(0));const depthValidity=Uint8Array.from(atob(data.depth_validity_base64),c=>c.charCodeAt(0));const depthBytes=Uint8Array.from(atob(data.depth_m_base64),c=>c.charCodeAt(0));const depthView=new DataView(depthBytes.buffer,depthBytes.byteOffset,depthBytes.byteLength);const hardCount=validity.reduce((a,b)=>a+b,0);const slider=document.getElementById('threshold');const inspector=document.getElementById('depthInspector');const marker=document.getElementById('depthMarker');let raf=0,currentCut=0,lastPixel=null;
function depthAt(index){return depthView.getFloat32(index*4,true)}
function update(){const t=Number(slider.value);currentCut=Math.round(t*255);document.getElementById('thresholdValue').textContent=t.toFixed(2);let shown=0;for(let i=0;i<validity.length;i++)if(validity[i]&&confidence[i]>=currentCut)shown++;document.getElementById('retainedHard').textContent=(hardCount?100*shown/hardCount:0).toFixed(1)+'%';document.getElementById('retainedFrame').textContent=(100*shown/validity.length).toFixed(1)+'%';if(lastPixel)showDepth(lastPixel[0],lastPixel[1]);cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>draw(currentCut));}
function draw(cut){document.querySelectorAll('canvas[data-mask]').forEach(canvas=>{const useDepth=canvas.dataset.mask.startsWith('depth'),mask=useDepth?depthValidity:validity;canvas.width=data.width;canvas.height=data.height;const ctx=canvas.getContext('2d'),image=ctx.createImageData(data.width,data.height);for(let i=0;i<mask.length;i++){let alpha=0,r=0,g=0,b=0;if(!mask[i]){alpha=((i%data.width+Math.floor(i/data.width))%8<3)?210:150;r=100;g=110;b=125}else if(confidence[i]<cut){alpha=225;r=2;g=6;b=23}const o=i*4;image.data[o]=r;image.data[o+1]=g;image.data[o+2]=b;image.data[o+3]=alpha}ctx.putImageData(image,0,0)});}
function showDepth(x,y){lastPixel=[x,y];const index=y*data.width+x,value=depthAt(index),isValid=depthValidity[index]&&confidence[index]>=currentCut&&Number.isFinite(value);marker.style.display='block';marker.style.left=((x+.5)/data.width*100)+'%';marker.style.top=((y+.5)/data.height*100)+'%';const readout=document.getElementById('depthReadout'),evidence=document.getElementById('depthEvidence');if(isValid){readout.textContent=value.toFixed(2)+' m';evidence.textContent='Confidence '+(confidence[index]/255).toFixed(2)+' · __DEPTH_TRUST_LABEL__ · pixel '+x+', '+y}else{readout.textContent='No valid depth';evidence.textContent=depthValidity[index]?'Hidden by confidence threshold':'No hard-valid correspondence at pixel '+x+', '+y}}
function inspect(event){const rect=inspector.getBoundingClientRect(),x=Math.max(0,Math.min(data.width-1,Math.floor((event.clientX-rect.left)/rect.width*data.width))),y=Math.max(0,Math.min(data.height-1,Math.floor((event.clientY-rect.top)/rect.height*data.height)));showDepth(x,y)}
const opacity=document.getElementById('overlayOpacity');function setOpacity(){const value=Number(opacity.value);document.getElementById('overlayOpacityValue').textContent=Math.round(value*100)+'%';document.getElementById('depthColorLayer').style.opacity=String(value);inspector.querySelector('canvas').style.opacity=String(value)}
inspector.addEventListener('pointermove',inspect);inspector.addEventListener('pointerdown',inspect);opacity.addEventListener('input',setOpacity);slider.addEventListener('input',update);document.getElementById('reset').onclick=()=>{slider.value='0';update()};setOpacity();update();
</script></body></html>"""
    replacements = {
        "__DISPOSITION__": html.escape(result.disposition.value),
        "__STATUS_CLASS__": "" if diagnostic else "trusted",
        "__ICON__": "⚠" if diagnostic else "✓",
        "__STATUS_TITLE__": html.escape(status_title),
        "__REASON__": html.escape(reason),
        "__ACTION__": html.escape(action),
        "__HARD_COVERAGE__": f"{hard_coverage:.1f}",
        "__DEPTH_NEAR__": f"{depth_near:.2f}" if np.isfinite(depth_near) else "n/a",
        "__DEPTH_FAR__": f"{depth_far:.2f}" if np.isfinite(depth_far) else "n/a",
        "__TRUST_COLOR__": "var(--rose)" if diagnostic else "var(--green)",
        "__TRUSTED__": "False" if diagnostic else "True",
        "__DEPTH_TRUST_LABEL__": (
            "diagnostic calibration" if diagnostic else "trusted depth"
        ),
        "__SOURCE_SECTION__": (
            '<section class="card source-section"><h2>Actual camera captures</h2>'
            '<p class="muted">The original uploaded images, before rectification or resizing.</p>'
            '<div class="source-grid"><div class="source-frame">'
            '<img src="source_main.png" alt="Actual Main camera capture">'
            '<div class="label"><b>Main</b><small>original upload</small></div></div>'
            '<div class="source-frame"><img src="source_ultrawide.png" '
            'alt="Actual Ultra-Wide camera capture"><div class="label">'
            '<b>Ultra-Wide</b><small>original upload</small></div></div></div></section>'
            if has_source_images
            else ""
        ),
    }
    for source, target in replacements.items():
        html_text = html_text.replace(source, target)
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_stereo_result(
    result: ProcessedStereoResult,
    output_dir: str | Path,
    *,
    main_input: str | None = None,
    ultrawide_input: str | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if result.disposition is ProcessingDisposition.REJECTED:
        reason = ", ".join(result.reason_codes) or "CALIBRATION_REJECTED"
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "disposition": result.disposition.value,
            "trusted_depth_eligible": False,
            "reason_codes": list(result.reason_codes),
            "recommended_action": result.calibration.recommended_action,
            "calibration": {
                "status": result.calibration.status.value,
                "fingerprint": result.calibration.fingerprint,
                "source": result.calibration.source_path,
            },
            "artifacts": {},
        }
        manifest_path = destination / "stereo_result.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary = {
            "status": result.disposition.value,
            "trusted_depth_eligible": False,
            "reason_codes": list(result.reason_codes),
            "manifest": str(manifest_path),
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        (destination / "index.html").write_text(
            "<!doctype html><meta name=viewport content='width=device-width'>"
            "<title>Stereo Processing Rejected</title>"
            "<h1>Stereo Processing Result: rejected</h1>"
            f"<p>Reason: <code>{html.escape(reason)}</code></p>"
            f"<p>{html.escape(result.calibration.recommended_action or '')}</p>",
            encoding="utf-8",
        )
        return manifest

    assert result.rectified is not None
    assert result.disparity is not None
    assert result.depth is not None
    disparity = result.disparity.left_to_right_px
    confidence = result.disparity.confidence
    validity = result.disparity.validity
    depth = result.depth.meters

    np.save(destination / "disparity_float32.npy", disparity.astype(np.float32))
    np.save(destination / "depth_m_float32.npy", depth.astype(np.float32))
    np.save(destination / "confidence_float32.npy", confidence.astype(np.float32))
    cv2.imwrite(
        str(destination / "confidence_u16.png"),
        np.rint(np.clip(confidence, 0, 1) * 65535).astype(np.uint16),
    )
    cv2.imwrite(
        str(destination / "disparity_validity.png"),
        validity.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(destination / "depth_validity.png"),
        result.depth.validity.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(destination / "rectified_ultrawide.png"),
        result.rectified.left_ultrawide,
    )
    cv2.imwrite(
        str(destination / "rectified_main.png"), result.rectified.right_main
    )
    disparity_color = _colorize(disparity, validity, cv2.COLORMAP_TURBO)
    depth_color = _colorize(
        depth, result.depth.validity, cv2.COLORMAP_PLASMA, reverse=True
    )
    cv2.imwrite(str(destination / "disparity_map.png"), disparity_color)
    cv2.imwrite(str(destination / "depth_map.png"), depth_color)
    overlay = cv2.addWeighted(
        result.rectified.left_ultrawide, 0.6, depth_color, 0.4, 0
    )
    cv2.imwrite(str(destination / "depth_overlay.png"), overlay)

    source_artifacts: list[str] = []
    if main_input and ultrawide_input:
        source_main = cv2.imread(str(main_input), cv2.IMREAD_COLOR)
        source_ultrawide = cv2.imread(str(ultrawide_input), cv2.IMREAD_COLOR)
        if source_main is not None and source_ultrawide is not None:
            cv2.imwrite(str(destination / "source_main.png"), source_main)
            cv2.imwrite(
                str(destination / "source_ultrawide.png"), source_ultrawide
            )
            source_artifacts = ["source_main.png", "source_ultrawide.png"]

    geometry = result.rectified.geometry
    files = [
        "disparity_float32.npy",
        "depth_m_float32.npy",
        "confidence_float32.npy",
        "confidence_u16.png",
        "disparity_validity.png",
        "depth_validity.png",
        "rectified_ultrawide.png",
        "rectified_main.png",
        "disparity_map.png",
        "depth_map.png",
        "depth_overlay.png",
    ] + source_artifacts
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "disposition": result.disposition.value,
        "trusted_depth_eligible": result.trusted_depth_eligible,
        "reason_codes": list(result.reason_codes),
        "recommended_action": result.calibration.recommended_action,
        "calibration": {
            "status": result.calibration.status.value,
            "fingerprint": result.calibration.fingerprint,
            "source": result.calibration.source_path,
        },
        "camera_order": {"left": "ultrawide", "right": "main"},
        "disparity_convention": "x_left_minus_x_right",
        "matcher": {
            "method": result.disparity.method.value,
            "profile_version": result.disparity.profile_version,
        },
        "geometry": {
            "focal_length_px": geometry.focal_length_px,
            "baseline_m": geometry.baseline_m,
            "disparity_offset_px": geometry.disparity_offset_px,
        },
        "coverage": {
            "rectification_joint": result.rectified.diagnostics[
                "joint_valid_fraction"
            ],
            "disparity_hard_valid": float(np.mean(validity)),
            "depth_valid": float(np.mean(result.depth.validity)),
        },
        "timings_ms": result.timings_ms,
        "inputs": {"main": main_input, "ultrawide": ultrawide_input},
        "artifacts": {},
    }
    for name in files:
        path = destination / name
        manifest["artifacts"][name] = {
            "path": name,
            "sha256": _sha256(path),
        }
    manifest_path = destination / "stereo_result.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "status": result.disposition.value,
        "trusted_depth_eligible": result.trusted_depth_eligible,
        "reason_codes": list(result.reason_codes),
        "manifest": str(manifest_path),
        "dashboard": str(destination / "index.html"),
        "rectified_focal_length_px": geometry.focal_length_px,
        "baseline_meters": geometry.baseline_m,
        "hard_valid_coverage": float(np.mean(validity)),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    confidence_u8 = np.rint(np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    _write_digest(
        destination,
        result,
        confidence_u8,
        has_source_images=bool(source_artifacts),
    )
    return manifest
