import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";

await tf.setBackend("cpu");

const W1 = tf.tensor2d([
  0.9,-0.2, 0.1, 0.4, 0.3, 0.2,-0.1, 0.6,
 -0.1, 0.7,-0.2, 0.3, 0.4,-0.1, 0.5,-0.3,
 -0.3,-0.1, 0.8,-0.4, 0.2, 0.6, 0.3, 0.1,
  0.2, 0.1,-0.3, 0.7,-0.2, 0.5, 0.2, 0.4
], [4,8]);
const b1 = tf.tensor1d([0.1, 0.0, 0.05, 0.08, -0.02, 0.03, 0.06, 0.01]);
const W2 = tf.tensor2d([
  0.4, 0.2,-0.1,
  0.6,-0.2,-0.1,
 -0.3, 0.5, 0.2,
  0.7, 0.1,-0.2,
 -0.1, 0.6, 0.4,
  0.2, 0.4, 0.3,
  0.5,-0.3, 0.2,
  0.3, 0.2, 0.5
], [8,3]);
const b2 = tf.tensor1d([0.05, 0.05, 0.1]);

const dense1 = tf.layers.dense({ units: 8, activation: "relu", inputShape: [4] });
const dense2 = tf.layers.dense({ units: 3, activation: "sigmoid" });
const model  = tf.sequential({ layers: [dense1, dense2] });
dense1.setWeights([W1, b1]);
dense2.setWeights([W2, b2]);

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const normalizeGenre = s => s.toLowerCase().replace(/\s+/g,'-').replace(/[^a-z0-9-]/g,'');
const banned = new Set(["reggaeton","latin","salsa","cumbia","regional-mexicano","banda","banda-ms","hip-hop","rap","trap","metal","punk"]);

function wmoToBucket(code){
  if (code === 0) return 0;
  if ([1,2,3].includes(code)) return 1;
  if ([45,48].includes(code)) return 2;
  if ((code>=51 && code<=67) || (code>=80 && code<=82)) return 3;
  if ((code>=71 && code<=77) || [85,86].includes(code)) return 4;
  if ([95,96,99].includes(code)) return 5;
  return 1;
}
function featuresFromWeather(wx){
  const tempC = wx.temperature;
  const tempNorm = (clamp(tempC, -10, 40) + 10) / 50;
  const bucket = wmoToBucket(wx.weathercode);
  const bucketNorm = bucket / 5;
  const isDay = wx.is_day ? 1 : 0;
  const hour = Number((wx.time || "00:00").split("T")[1]?.slice(0,2) || 0);
  const hourNorm = hour / 23;
  return { tempC, tempNorm, bucket, bucketNorm, isDay, hour, hourNorm };
}
function inferMood(f){
  const x = tf.tensor2d([[f.tempNorm, f.hourNorm, f.bucketNorm, f.isDay]]);
  const [energy, valence, acousticness] = model.predict(x).dataSync();
  x.dispose();
  return { energy, valence, acousticness };
}

function dayOfYear(iso){
  const d = new Date(iso || Date.now());
  const start = new Date(Date.UTC(d.getUTCFullYear(),0,1));
  const diff = (d - start) / 86400000;
  return Math.floor(diff)+1;
}
function rngSeed(s){ let x = s|0; return () => { x ^= x<<13; x ^= x>>>17; x ^= x<<5; return ((x>>>0)/4294967296); }; }
function pickWeighted(rand, items){
  const total = items.reduce((a,b)=>a+b.w,0);
  let r = rand()*total;
  for(const it of items){ if((r-=it.w)<=0) return it.k; }
  return items[items.length-1].k;
}

const FAMILIES = {
  chill: ["chill","lofi","chillhop","ambient","downtempo","electronic ambient"],
  jazz: ["jazz","nu jazz","bossa nova","jazz instrumental","blues"],
  brightpop: ["indie pop","synthpop","electropop","dream pop","pop"],
  groove: ["funk","nu-disco","disco","house","deep house","electronica"],
  acoustic: ["acoustic","folk","singer-songwriter","piano"],
  soul: ["soul","neo soul","r&b"]
};
const FAMILY_SEEDS = {
  chill: ["chill","ambient","electronic","downtempo","lo-fi"],
  jazz: ["jazz","bossanova","jazztronica","blues"],
  brightpop: ["indie-pop","pop","electropop","synth-pop","dance-pop"],
  groove: ["funk","disco","house","deep-house","electronic"],
  acoustic: ["acoustic","folk","singer-songwriter","piano","classical"],
  soul: ["soul","r-n-b","neo-soul","blues"]
};
for (const fam in FAMILY_SEEDS){
  FAMILY_SEEDS[fam] = FAMILY_SEEDS[fam].map(normalizeGenre).filter(g=>!banned.has(g));
}

function contextWeights(ctx, wx, mood){
  const w = { chill:1, jazz:1, brightpop:1, groove:1, acoustic:1, soul:1 };
  const hour = ctx.hour;
  const bucket = wx.bucket;
  const weekend = ctx.weekend;
  if (hour>=0 && hour<6) { w.chill+=2; w.acoustic+=0.5; w.jazz+=0.5; w.brightpop-=0.5; w.groove-=0.5; }
  else if (hour>=6 && hour<12) { w.brightpop+=2; w.acoustic+=1; w.chill+=0.5; }
  else if (hour>=12 && hour<18) { w.groove+=1.5; w.brightpop+=1; w.soul+=0.5; }
  else { w.jazz+=1.5; w.soul+=1; w.chill+=0.5; }
  if (bucket===3 || bucket===2) { w.chill+=1.5; w.jazz+=0.5; w.brightpop-=0.5; w.groove-=0.5; }
  if (bucket===0 && hour>=19) { w.jazz+=1.5; w.soul+=0.5; }
  if (bucket===5) { w.chill+=1; w.groove+=0.5; }
  if (weekend) { w.groove+=0.8; w.brightpop+=0.6; w.chill+=0.2; }
  if (mood.energy<0.45) { w.chill+=0.5; w.acoustic+=0.5; w.jazz+=0.3; }
  if (mood.energy>0.7 && bucket!==3 && bucket!==2) { w.groove+=0.6; w.brightpop+=0.4; }
  for (const k in w){ if (w[k]<0.2) w[k]=0.2; }
  return w;
}

function buildTerms(rand, fam, neighbor){
  const base = FAMILIES[fam] || [];
  const near = FAMILIES[neighbor] || [];
  const pool = [...new Set([...base, ...(rand()<0.35?near:[])])];
  const pick = (arr) => arr[Math.floor(rand()*arr.length)];
  const t1 = pick(pool);
  let t2 = pick(pool.filter(x=>x!==t1));
  if (!t2) t2 = t1;
  return [t1, t2];
}
function seedTerms(seeds = []) {
  const map = { "alt-rock":"alternative rock","bossanova":"bossa nova","r-n-b":"r&b","hip-hop":"hip hop","lo-fi":"lofi","synth-pop":"synthpop","deep-house":"deep house","indie-pop":"indie pop" };
  return Array.from(new Set((seeds || []).map(s => (map[s] || s).replace(/-/g, " ")).filter(Boolean)));
}
function yearWindowForFamily(fam){
  const y = new Date().getUTCFullYear();
  if (fam==="jazz" || fam==="soul") return `${y-60}-${y}`;
  if (fam==="acoustic") return `${y-35}-${y}`;
  if (fam==="chill") return `${y-20}-${y}`;
  if (fam==="groove" || fam==="brightpop") return `${y-12}-${y}`;
  return `${y-20}-${y}`;
}
function jitter(rand, v, r){ return clamp(v + (rand()*2-1)*r, 0, 1); }

function quantize(n, step = 0.05){ return Math.round(n/step)*step; }
const WX_TTL_MS = 5 * 60 * 1000;
const wxMemCache = new Map();
function wxCacheKey(lat, lon){ return `${quantize(lat,0.05).toFixed(2)},${quantize(lon,0.05).toFixed(2)}`; }
function wxCacheGet(k){ const e = wxMemCache.get(k); return e && (Date.now()-e.ts) < WX_TTL_MS ? e.data : null; }
function wxCachePut(k, d){ wxMemCache.set(k, { ts: Date.now(), data: d }); }
async function fetchWithBackoff(url, init, attempts = 3){
  for (let i=0;i<attempts;i++){
    const r = await fetch(url, init);
    if (r.status !== 429 || i === attempts-1) return r;
    const wait = (2**i)*500 + Math.random()*250;
    await new Promise(res => setTimeout(res, wait));
  }
}
async function fetchWeatherOpenMeteo(env, lat, lon){
  const url = new URL(env.OPEN_METEO_URL);
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("current_weather", "true");
  url.searchParams.set("timezone", "auto");
  const r = await fetchWithBackoff(url.toString());
  if (!r.ok) throw new Error(`Open-Meteo: ${r.status} ${await r.text()}`);
  const j = await r.json();
  return j.current_weather;
}
function symbolToWmoLike(symbol){
  const s = String(symbol || "").toLowerCase();
  if (s.includes("thunder")) return 95;
  if (s.includes("snow") || s.includes("sleet")) return 75;
  if (s.includes("rain")) return 60;
  if (s.includes("fog")) return 45;
  if (s.includes("clearsky")) return 0;
  return 3;
}
async function fetchWeatherMetNo(env, lat, lon){
  const url = `https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=${lat}&lon=${lon}`;
  const ua = env.METNO_UA || "no24.app/1.0 (contact: example@example.com)";
  const r = await fetch(url, { headers: { "User-Agent": ua } });
  if (!r.ok) throw new Error(`METNO: ${r.status} ${await r.text()}`);
  const j = await r.json();
  const ts = j.properties?.timeseries?.[0];
  if (!ts) throw new Error("METNO: empty timeseries");
  const temp = ts.data?.instant?.details?.air_temperature;
  const symbol = ts.data?.next_1_hours?.summary?.symbol_code || ts.data?.next_6_hours?.summary?.symbol_code || "";
  const weathercode = symbolToWmoLike(symbol);
  const iso = ts.time || "";
  const hour = Number(iso.slice(11,13) || 0);
  const is_day = hour >= 7 && hour < 19 ? 1 : 0;
  return { temperature: temp, weathercode, is_day, time: iso };
}
async function fetchWeather(env, lat, lon){
  const key = wxCacheKey(lat, lon);
  const c = wxCacheGet(key);
  if (c) return c;
  try {
    const cw = await fetchWeatherOpenMeteo(env, lat, lon);
    wxCachePut(key, cw);
    return cw;
  } catch(_) {}
  const cw2 = await fetchWeatherMetNo(env, lat, lon);
  wxCachePut(key, cw2);
  return cw2;
}

async function reverseGeocode(env, lat, lon){
  const url = new URL(env.OPEN_METEO_GEO_URL);
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("language", "es");
  url.searchParams.set("format", "json");
  const r = await fetch(url.toString());
  if (!r.ok) return null;
  const j = await r.json();
  const first = j.results?.[0];
  if (!first) return null;
  return {
    name: first.name || "",
    admin1: first.admin1 || "",
    country_code: first.country_code || ""
  };
}

async function getSpotifyToken(env){
  if (!env.SPOTIFY_CLIENT_ID || !env.SPOTIFY_CLIENT_SECRET) {
    throw new Error("SPOTIFY_SECRETS_MISSING: set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET");
  }
  if (globalThis.__spToken && Date.now() < globalThis.__spToken.exp) return globalThis.__spToken.token;
  const creds = btoa(`${env.SPOTIFY_CLIENT_ID}:${env.SPOTIFY_CLIENT_SECRET}`);
  const res = await fetch("https://accounts.spotify.com/api/token", {
    method: "POST",
    headers: { "Authorization": `Basic ${creds}`, "Content-Type": "application/x-www-form-urlencoded" },
    body: "grant_type=client_credentials"
  });
  const txt = await res.text();
  if (!res.ok) throw new Error(`SPOTIFY_TOKEN_FAILED ${res.status}: ${txt}`);
  const data = JSON.parse(txt);
  globalThis.__spToken = { token: data.access_token, exp: Date.now() + (data.expires_in - 60) * 1000 };
  return data.access_token;
}
async function getAvailableGenres(env){
  if (globalThis.__genres && (Date.now() - globalThis.__genres.ts) < 3600_000) return globalThis.__genres.list;
  let list = null;
  try {
    const token = await getSpotifyToken(env);
    const r = await fetch("https://api.spotify.com/v1/recommendations/available-genre-seeds", { headers: { Authorization: `Bearer ${token}` } });
    if (r.ok) { const j = await r.json(); list = new Set((j.genres || []).filter(g=>!banned.has(g))); }
  } catch {}
  if (!list) list = new Set(["pop","rock","dance","edm","indie","alt-rock","jazz","bossanova","blues","ambient","classical","acoustic","folk","piano","r-n-b","soul","electronic","house","deep-house","disco","nu-disco","electropop","synth-pop"].filter(g=>!banned.has(g)));
  globalThis.__genres = { list, ts: Date.now() };
  return list;
}

async function searchTrackByMood(env, rand, mood, terms, market = "US", yearRange = null){
  const token = await getSpotifyToken(env);
  const yearNow = new Date().getUTCFullYear();
  const range = yearRange || `${yearNow-10}-${yearNow}`;
  const quoted = terms.map(t => `"${t}"`).join(" OR ");
  const q = `(${quoted}) year:${range}`;
  const params = new URLSearchParams({ q, type: "track", market, limit: "30" });
  const r = await fetch(`https://api.spotify.com/v1/search?${params}`, { headers: { Authorization: `Bearer ${token}` } });
  const txt = await r.text();
  if (!r.ok) throw new Error(`SPOTIFY_SEARCH_FAILED ${r.status}: ${txt}`);
  const j = JSON.parse(txt);
  let tracks = j.tracks?.items || [];
  if (!tracks.length) {
    const q2 = `(${quoted})`;
    const r2 = await fetch(`https://api.spotify.com/v1/search?${new URLSearchParams({ q: q2, type: "track", market, limit:"30" })}`, { headers: { Authorization: `Bearer ${token}` } });
    const t2 = await r2.text();
    if (!r2.ok) throw new Error(`SPOTIFY_SEARCH_FAILED ${r2.status}: ${t2}`);
    const j2 = JSON.parse(t2);
    tracks = j2.tracks?.items || [];
  }
  if (!tracks.length) return null;

  try {
    const ids = tracks.slice(0, 50).map(t => t.id).join(",");
    const rf = await fetch(`https://api.spotify.com/v1/audio-features?ids=${ids}`, { headers: { Authorization: `Bearer ${token}` } });
    if (rf.ok) {
      const jf = await rf.json();
      const feats = jf.audio_features || [];
      let best = null, bestScore = Infinity;
      const targetDance = (mood.energy + mood.valence)/2;
      for (let i = 0; i < feats.length; i++) {
        const f = feats[i]; if (!f) continue;
        const dEnergy = (f.energy - mood.energy);
        const dVal = (f.valence - mood.valence);
        const dDance = ((f.danceability ?? 0.5) - targetDance);
        const eps = (rand()*0.15);
        const score = Math.hypot(dEnergy, dVal, dDance) - eps;
        if (score < bestScore) { bestScore = score; best = tracks[i]; }
      }
      if (best) return { id: best.id, name: best.name, artists: (best.artists || []).map(a => a.name).join(", "), url: `https://open.spotify.com/track/${best.id}`, preview_url: best.preview_url };
    }
  } catch {}

  const t = tracks[Math.floor(rand()*Math.min(tracks.length, 10))];
  return { id: t.id, name: t.name, artists: (t.artists || []).map(a => a.name).join(", "), url: `https://open.spotify.com/track/${t.id}`, preview_url: t.preview_url };
}

async function recommendTrack(env, rand, mood, fam, terms, market = "US"){
  try {
    const avail = await getAvailableGenres(env);
    const seeds = (FAMILY_SEEDS[fam] || []).filter(g => avail.has(g));
    if (seeds.length) {
      const seedGenres = Array.from(new Set(seeds)).slice(0,5).join(",");
      const token = await getSpotifyToken(env);
      const params = new URLSearchParams({
        limit: "1",
        seed_genres: seedGenres,
        target_energy: jitter(rand, mood.energy, 0.15).toFixed(2),
        target_valence: jitter(rand, mood.valence, 0.15).toFixed(2),
        min_popularity: "40",
        market
      });
      const url = `https://api.spotify.com/v1/recommendations?${params}`;
      const r = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
      const txt = await r.text();
      if (r.status !== 404 && !r.ok) throw new Error(`SPOTIFY_RECS_FAILED ${r.status}: ${txt}`);
      if (r.ok) {
        const j = JSON.parse(txt);
        const t = j.tracks?.[0];
        if (t) return { id: t.id, name: t.name, artists: (t.artists || []).map(a => a.name).join(", "), url: `https://open.spotify.com/track/${t.id}`, preview_url: t.preview_url };
      }
    }
  } catch {}
  const yr = yearWindowForFamily(fam);
  return await searchTrackByMood(env, rand, mood, terms, market, yr);
}

function json(data, init = {}){
  return new Response(JSON.stringify(data, null, 2), { headers: { "content-type": "application/json; charset=utf-8","access-control-allow-origin": "*" }, ...init });
}

function html(){
  return new Response(`<!doctype html>
<html lang="es">
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>no24</title>
<style>
  :root { color-scheme: dark; }
  html,body{height:100%}
  body{margin:0;background:#000;color:#fff;display:grid;place-items:center;font: clamp(18px, 2.5vw, 28px)/1.3 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;}
  .center{ text-align:center; }
  .step{ font-weight:600; opacity:0; animation: fadeIn 400ms ease forwards; }
  .info{ margin-top:12px; min-height:1.5em; }
  .bubble{ display:inline-block; opacity:0; transform:translateY(8px); }
  .fade { animation: infoFade 1400ms ease forwards; }
  @keyframes fadeIn{ from{opacity:0} to{opacity:1} }
  @keyframes infoFade{
    0%{opacity:0; transform:translateY(8px)}
    15%{opacity:1; transform:translateY(0)}
    80%{opacity:1}
    100%{opacity:0; transform:translateY(-6px)}
  }
</style>
<div class="center">
  <div id="step" class="step">1. Analizando clima</div>
  <div id="info" class="info"></div>
</div>
<script>
(function(){
  var $ = function(s){ return document.querySelector(s); };
  var step = $('#step');
  var info = $('#info');
  function setStep(t){ step.textContent = t; step.classList.remove('step'); void step.offsetWidth; step.classList.add('step'); }
  function sleep(ms){ return new Promise(function(r){ setTimeout(r, ms); }); }
  function showOnce(txt, dur){
    return new Promise(function(resolve){
      info.innerHTML = '';
      var span = document.createElement('span');
      span.className = 'bubble fade';
      span.textContent = txt;
      info.appendChild(span);
      setTimeout(resolve, dur||1400);
    });
  }
  function formatLocal(dtISO, tz){
    try {
      var d = new Date(dtISO);
      var day = new Intl.DateTimeFormat('es-MX',{weekday:'long', timeZone: tz}).format(d);
      var time = new Intl.DateTimeFormat('es-MX',{hour:'2-digit', minute:'2-digit', hour12:false, timeZone: tz}).format(d);
      return { day: day.charAt(0).toUpperCase()+day.slice(1), time };
    } catch(e){
      var d2 = new Date();
      var day2 = new Intl.DateTimeFormat('es-MX',{weekday:'long'}).format(d2);
      var time2 = new Intl.DateTimeFormat('es-MX',{hour:'2-digit', minute:'2-digit', hour12:false}).format(d2);
      return { day: day2.charAt(0).toUpperCase()+day2.slice(1), time: time2 };
    }
  }
  function getPos(){
    return new Promise(function(resolve, reject){
      navigator.geolocation.getCurrentPosition(
        function(p){ resolve({ lat: p.coords.latitude, lon: p.coords.longitude }); },
        function(e){ reject(e); },
        { enableHighAccuracy: true, timeout: 8000 }
      );
    });
  }
  (async function run(){
    try{
      setStep('1. Analizando clima');
      var pos = await getPos();

      var uMeta = new URL(location.href); uMeta.pathname = '/api/meta';
      uMeta.search = new URLSearchParams({ lat: pos.lat, lon: pos.lon }).toString();
      var rMeta = await fetch(uMeta.toString());
      var m = await rMeta.json();

      var loc = m && m.city ? m.city : 'Tu ubicación';
      var tC = m && typeof m.temperature === 'number' ? Math.round(m.temperature) + '°C' : '';
      var local = formatLocal(m && m.time ? m.time : null, m && m.timezone ? m.timezone : undefined);

      await showOnce('Ciudad: ' + loc, 1400);
      await showOnce('Temperatura: ' + tC, 1400);
      await showOnce(local.day + ' • ' + local.time, 1400);

      setStep('2. Consiguiendo una canción');
      var u = new URL(location.href);
      u.pathname = '/api/track';
      u.search = new URLSearchParams({ lat: pos.lat, lon: pos.lon }).toString();
      var r = await fetch(u.toString());
      var j = await r.json();
      if(!r.ok || !j || !j.track || !j.track.url) throw new Error(j && j.error || 'Sin canción');

      for (var n = 3; n >= 1; n--) {
        setStep('3. Abriendo Spotify en ' + n);
        await sleep(1000);
      }
      location.replace(j.track.url);
    } catch(e){
      step.textContent = 'No pude obtener ubicación o canción. Revisa permisos y recarga.';
      info.textContent = '';
    }
  })();
})();
</script>
</html>`, { headers: { "content-type": "text/html; charset=utf-8" } });
}

export default {
  async fetch(req, env){
    const url = new URL(req.url);
    const { pathname, searchParams } = url;
    const cfCountry = req.cf?.country;
    const market = /^[A-Z]{2}$/.test(cfCountry || '') ? cfCountry : 'US';

    if (pathname === "/") return html();

    if (pathname === "/api/ping"){
      try {
        const lat = Number(searchParams.get("lat") || 0);
        const lon = Number(searchParams.get("lon") || 0);
        const cw = (Number.isFinite(lat) && Number.isFinite(lon)) ? await fetchWeather(env, lat, lon) : { ok: true };
        return json({ ok: true, weather: cw });
      } catch (e) { return json({ ok: false, error: String(e?.message || e) }, { status: 500 }); }
    }

    if (pathname === "/api/meta"){
      try{
        const lat = Number(searchParams.get("lat"));
        const lon = Number(searchParams.get("lon"));
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return json({ error: "lat/lon requeridos" }, { status: 400 });
        const cw = await fetchWeather(env, lat, lon);
        const rg = await reverseGeocode(env, lat, lon);
        const city = rg ? (rg.name + (rg.admin1?(", "+rg.admin1):"") + (rg.country_code?(" • "+rg.country_code):"")) : null;
        const tz = req.cf?.timezone || "UTC";
        return json({ city, temperature: cw.temperature, is_day: cw.is_day, weathercode: cw.weathercode, time: cw.time, timezone: tz });
      } catch(e){
        return json({ error: String(e?.message || e) }, { status: 500 });
      }
    }

    if (pathname === "/api/track"){
      const debug = searchParams.get("debug") === "1";
      try{
        const lat = Number(searchParams.get("lat"));
        const lon = Number(searchParams.get("lon"));
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return json({ error: "lat/lon requeridos" }, { status: 400 });

        const cw   = await fetchWeather(env, lat, lon);
        const feat = featuresFromWeather(cw);
        const mood0 = inferMood(feat);

        const doy = dayOfYear(cw.time);
        const seedVal = (Date.now() ^ Math.floor(lat*1e4) ^ Math.floor(lon*1e4) ^ (feat.hour<<8) ^ (doy<<16)) | 0;
        const rand = rngSeed(seedVal);

        let mood = { ...mood0 };
        if (feat.bucket===3 || feat.bucket===2) { mood.energy = clamp(mood.energy-0.1,0,1); mood.valence = clamp(mood.valence-0.05,0,1); }
        if (feat.hour>=6 && feat.hour<12) { mood.valence = clamp(mood.valence+0.1,0,1); }
        if (feat.hour>=18 && feat.hour<24) { mood.energy = clamp(mood.energy-0.05,0,1); }

        const ctx = { hour: feat.hour, weekend: [5,6,0].includes(new Date(cw.time).getUTCDay()), lat, lon };
        const weights = contextWeights(ctx, feat, mood);
        const fam = pickWeighted(rand, Object.keys(weights).map(k=>({k,w:Math.max(0.1,weights[k])})));

        const fams = ["chill","jazz","brightpop","groove","acoustic","soul"];
        let neighbor = fams.filter(x=>x!==fam)[Math.floor(rand()*(fams.length-1))];
        const terms = buildTerms(rand, fam, neighbor);

        const seeds = (FAMILY_SEEDS[fam] || []);
        const cleanSeeds = Array.from(new Set(seeds.map(normalizeGenre).filter(g=>!banned.has(g))));

        const track = await recommendTrack(env, rand, mood, fam, seedTerms(cleanSeeds.length?cleanSeeds:terms), market);
        if (!track) return json({ error: "Sin recomendaciones", weather: cw, features: mood, family: fam, terms, seeds: cleanSeeds }, { status: 502 });

        const buckets = ["despejado","nublado","niebla","lluvia","nieve","tormenta"];
        const summary = buckets[feat.bucket] + " \u2022 " + (cw.is_day ? "día" : "noche");

        const payload = { weather: { ...cw, summary }, context: { family: fam, terms, seeds: cleanSeeds, market }, features: mood, track };
        return json(debug ? { debug: true, ...payload } : payload);
      } catch (err){
        return json({ error: String(err?.message || err), stack: debug ? (err?.stack || null) : undefined }, { status: 500 });
      }
    }

    if (pathname === "/go"){
      try{
        const lat = Number(searchParams.get("lat"));
        const lon = Number(searchParams.get("lon"));
        const cw   = await fetchWeather(env, lat, lon);
        const feat = featuresFromWeather(cw);
        const mood0 = inferMood(feat);
        const doy = dayOfYear(cw.time);
        const seedVal = (Date.now() ^ Math.floor(lat*1e4) ^ Math.floor(lon*1e4) ^ (feat.hour<<8) ^ (doy<<16)) | 0;
        const rand = rngSeed(seedVal);
        let mood = { ...mood0 };
        if (feat.bucket===3 || feat.bucket===2) { mood.energy = clamp(mood.energy-0.1,0,1); mood.valence = clamp(mood.valence-0.05,0,1); }
        if (feat.hour>=6 && feat.hour<12) { mood.valence = clamp(mood.valence+0.1,0,1); }
        if (feat.hour>=18 && feat.hour<24) { mood.energy = clamp(mood.energy-0.05,0,1); }
        const ctx = { hour: feat.hour, weekend: [5,6,0].includes(new Date(cw.time).getUTCDay()), lat, lon };
        const weights = contextWeights(ctx, feat, mood);
        const fams = Object.keys(weights);
        const fam = pickWeighted(rand, fams.map(k=>({k,w:Math.max(0.1,weights[k])})));
        const terms = buildTerms(rand, fam, "jazz");
        const seeds = (FAMILY_SEEDS[fam] || []);
        const cleanSeeds = Array.from(new Set(seeds.map(normalizeGenre).filter(g=>!banned.has(g))));
        const track = await recommendTrack(env, rand, mood, fam, seedTerms(cleanSeeds.length?cleanSeeds:terms), /^[A-Z]{2}$/.test(req.cf?.country||"")?req.cf.country:"US");
        const loc = track ? track.url : "https://open.spotify.com/";
        return Response.redirect(loc, 302);
      } catch { return Response.redirect("https://open.spotify.com/", 302); }
    }

    return new Response("Not found", { status: 404 });
  }
};