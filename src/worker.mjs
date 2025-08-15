import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
await tf.setBackend("cpu");

const W1=tf.tensor2d([0.9,-0.2,0.1,0.4,0.3,0.2,-0.1,0.6,-0.1,0.7,-0.2,0.3,0.4,-0.1,0.5,-0.3,-0.3,-0.1,0.8,-0.4,0.2,0.6,0.3,0.1,0.2,0.1,-0.3,0.7,-0.2,0.5,0.2,0.4],[4,8]);
const b1=tf.tensor1d([0.1,0,0.05,0.08,-0.02,0.03,0.06,0.01]);
const W2=tf.tensor2d([0.4,0.2,-0.1,0.6,-0.2,-0.1,-0.3,0.5,0.2,0.7,0.1,-0.2,-0.1,0.6,0.4,0.2,0.4,0.3,0.5,-0.3,0.2,0.3,0.2,0.5],[8,3]);
const b2=tf.tensor1d([0.05,0.05,0.1]);
const dense1=tf.layers.dense({units:8,activation:"relu",inputShape:[4]});
const dense2=tf.layers.dense({units:3,activation:"sigmoid"});
const model=tf.sequential({layers:[dense1,dense2]});
dense1.setWeights([W1,b1]);dense2.setWeights([W2,b2]);

const clamp=(v,min,max)=>Math.min(max,Math.max(min,v));
const normGenre=s=>s.toLowerCase().replace(/\s+/g,"-").replace(/[^a-z0-9-]/g,"");
const banned=new Set(["reggaeton","latin","salsa","cumbia","regional-mexicano","banda","banda-ms","hip-hop","rap","trap","metal","punk"]);
const mem=new Map();

function recKey(lat,lon){const a=Math.floor(lat*20)/20,b=Math.floor(lon*20)/20;return`${a.toFixed(2)},${b.toFixed(2)}`}
function recGet(k){const e=mem.get(k);if(!e||(Date.now()-e.ts)>21600000)return{tids:new Set(),aids:new Set(),an:new Set()};return{tids:new Set(e.tids||[]),aids:new Set(e.aids||[]),an:new Set(e.an||[])}}
function recPush(k,t){const e=mem.get(k)||{tids:[],aids:[],an:[],ts:Date.now()};if(t?.id)e.tids.unshift(t.id);if(Array.isArray(t?.artist_ids))for(const id of t.artist_ids)if(id)e.aids.unshift(id);if(t?.artists)String(t.artists).split(",").map(x=>x.trim().toLowerCase()).filter(Boolean).forEach(n=>e.an.unshift(n));e.tids=Array.from(new Set(e.tids)).slice(0,60);e.aids=Array.from(new Set(e.aids)).slice(0,80);e.an=Array.from(new Set(e.an)).slice(0,80);e.ts=Date.now();mem.set(k,e)}
function readCookie(req){const raw=req.headers.get("Cookie")||"";const m=raw.match(/(?:^|;\s*)n24r=([^;]+)/);if(!m)return{tids:new Set(),aids:new Set(),an:new Set()};try{const val=decodeURIComponent(m[1]);const str=atob(val.replace(/-/g,"+").replace(/_/g,"/"));const obj=JSON.parse(str);if(!obj||typeof obj.ts!=="number"||(Date.now()-obj.ts)>21600000)return{tids:new Set(),aids:new Set(),an:new Set()};return{tids:new Set(obj.tids||[]),aids:new Set(obj.aids||[]),an:new Set(obj.an||[])}}catch{return{tids:new Set(),aids:new Set(),an:new Set()}}}
function writeCookie(rec){const out={ts:Date.now(),tids:Array.from(rec.tids).slice(0,20),aids:Array.from(rec.aids).slice(0,30),an:Array.from(rec.an).slice(0,30)};const s=btoa(JSON.stringify(out)).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");return`n24r=${encodeURIComponent(s)}; Max-Age=21600; Path=/; SameSite=Lax; Secure`}

function wmoBucket(c){if(c===0)return 0;if([1,2,3].includes(c))return 1;if([45,48].includes(c))return 2;if((c>=51&&c<=67)||(c>=80&&c<=82))return 3;if((c>=71&&c<=77)||[85,86].includes(c))return 4;if([95,96,99].includes(c))return 5;return 1}
function featFromWx(wx){const t=wx.temperature;const tn=(clamp(t,-10,40)+10)/50;const b=wmoBucket(wx.weathercode);const bn=b/5;const d=wx.is_day?1:0;const h=Number((wx.time||"00:00").split("T")[1]?.slice(0,2)||0);const hn=h/23;return{tempC:t,tempNorm:tn,bucket:b,bucketNorm:bn,isDay:d,hour:h,hourNorm:hn,timeISO:wx.time}}
function inferMood(f){const x=tf.tensor2d([[f.tempNorm,f.hourNorm,f.bucketNorm,f.isDay]]);const[a,b,c]=model.predict(x).dataSync();x.dispose();return{energy:a,valence:b,acousticness:c}}
function dayOfYear(iso){const d=new Date(iso||Date.now());const s=new Date(Date.UTC(d.getUTCFullYear(),0,1));return Math.floor((d-s)/86400000)+1}
function rngSeed(s){let x=s|0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return((x>>>0)/4294967296)}}
function pickWeighted(r,it){const tot=it.reduce((a,b)=>a+b.w,0);let v=r()*tot;for(const z of it){if((v-=z.w)<=0)return z.k}return it[it.length-1].k}

const FAMILIES={chill:["chill","lofi","chillhop","ambient","downtempo","electronic ambient"],jazz:["jazz","nu jazz","bossa nova","jazz instrumental","blues"],brightpop:["indie pop","synthpop","electropop","dream pop","pop"],groove:["funk","nu-disco","disco","house","deep house","electronica"],acoustic:["acoustic","folk","singer-songwriter","piano"],soul:["soul","neo soul","r&b"]};
const FAMILY_SEEDS={chill:["chill","ambient","electronic","downtempo","lo-fi"],jazz:["jazz","bossanova","jazztronica","blues"],brightpop:["indie-pop","pop","electropop","synth-pop","dance-pop"],groove:["funk","disco","house","deep-house","electronic"],acoustic:["acoustic","folk","singer-songwriter","piano","classical"],soul:["soul","r-n-b","neo-soul","blues"]};
for(const k in FAMILY_SEEDS){FAMILY_SEEDS[k]=FAMILY_SEEDS[k].map(normGenre).filter(g=>!banned.has(g))}

function ctxWeights(ctx,wx,m){const w={chill:1,jazz:1,brightpop:1,groove:1,acoustic:1,soul:1};const h=ctx.hour;const b=wx.bucket;const we=ctx.weekend;if(h>=0&&h<6){w.chill+=2;w.acoustic+=0.5;w.jazz+=0.5;w.brightpop-=0.5;w.groove-=0.5}else if(h>=6&&h<12){w.brightpop+=2;w.acoustic+=1;w.chill+=0.5}else if(h>=12&&h<18){w.groove+=1.5;w.brightpop+=1;w.soul+=0.5}else{w.jazz+=1.5;w.soul+=1;w.chill+=0.5}if(b===3||b===2){w.chill+=1.5;w.jazz+=0.5;w.brightpop-=0.5;w.groove-=0.5}if(b===0&&h>=19){w.jazz+=1.5;w.soul+=0.5}if(b===5){w.chill+=1;w.groove+=0.5}if(we){w.groove+=0.8;w.brightpop+=0.6;w.chill+=0.2}if(m.energy<0.45){w.chill+=0.5;w.acoustic+=0.5;w.jazz+=0.3}if(m.energy>0.7&&b!==3&&b!==2){w.groove+=0.6;w.brightpop+=0.4}for(const q in w){if(w[q]<0.2)w[q]=0.2}return w}

function buildTerms(r,fam,nei){const base=FAMILIES[fam]||[];const near=FAMILIES[nei]||[];const pool=[...new Set([...base,...(r()<0.35?near:[])])];const pick=a=>a[Math.floor(r()*a.length)];const t1=pick(pool);let t2=pick(pool.filter(x=>x!==t1));if(!t2)t2=t1;return[t1,t2]}
function seedTerms(seeds=[]){const map={"alt-rock":"alternative rock",bossanova:"bossa nova","r-n-b":"r&b","hip-hop":"hip hop","lo-fi":"lofi","synth-pop":"synthpop","deep-house":"deep house","indie-pop":"indie pop"};return Array.from(new Set((seeds||[]).map(s=>(map[s]||s).replace(/-/g," ")).filter(Boolean)))}

function q(n,step=0.05){return Math.round(n/step)*step}
const WX_TTL=300000;const wxCache=new Map();
function wxKey(a,b){return`${q(a,0.05).toFixed(2)},${q(b,0.05).toFixed(2)}`}
function wxGet(k){const e=wxCache.get(k);return e&&(Date.now()-e.ts)<WX_TTL?e.data:null}
function wxPut(k,d){wxCache.set(k,{ts:Date.now(),data:d})}
async function backoff(url,init,a=3){for(let i=0;i<a;i++){const r=await fetch(url,init);if(r.status!==429||i===a-1)return r;const w=(2**i)*500+Math.random()*250;await new Promise(res=>setTimeout(res,w))}}
async function wxOM(env,lat,lon){const u=new URL(env.OPEN_METEO_URL);u.searchParams.set("latitude",String(lat));u.searchParams.set("longitude",String(lon));u.searchParams.set("current_weather","true");u.searchParams.set("timezone","auto");const r=await backoff(u.toString());if(!r.ok)throw new Error(`Open-Meteo: ${r.status} ${await r.text()}`);const j=await r.json();return j.current_weather}
function symToWMO(s){const x=String(s||"").toLowerCase();if(x.includes("thunder"))return 95;if(x.includes("snow")||x.includes("sleet"))return 75;if(x.includes("rain"))return 60;if(x.includes("fog"))return 45;if(x.includes("clearsky"))return 0;return 3}
async function wxMETNO(env,lat,lon){const u=`https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=${lat}&lon=${lon}`;const ua=env.METNO_UA||"no24.app/1.0 (contact: example@example.com)";const r=await fetch(u,{headers:{"User-Agent":ua}});if(!r.ok)throw new Error(`METNO: ${r.status} ${await r.text()}`);const j=await r.json();const ts=j.properties?.timeseries?.[0];if(!ts)throw new Error("METNO: empty");const t=ts.data?.instant?.details?.air_temperature;const sym=ts.data?.next_1_hours?.summary?.symbol_code||ts.data?.next_6_hours?.summary?.symbol_code||"";const w=symToWMO(sym);const iso=ts.time||"";const h=Number(iso.slice(11,13)||0);const isd=h>=7&&h<19?1:0;return{temperature:t,weathercode:w,is_day:isd,time:iso}}
async function fetchWeather(env,lat,lon){const k=wxKey(lat,lon);const c=wxGet(k);if(c)return c;try{const a=await wxOM(env,lat,lon);wxPut(k,a);return a}catch{}const b=await wxMETNO(env,lat,lon);wxPut(k,b);return b}
async function reverseGeocode(env,lat,lon){const u=new URL(env.OPEN_METEO_GEO_URL||"https://geocoding-api.open-meteo.com/v1/reverse");u.searchParams.set("latitude",String(lat));u.searchParams.set("longitude",String(lon));u.searchParams.set("language","es");u.searchParams.set("format","json");const r=await fetch(u.toString());if(!r.ok)return null;const j=await r.json();const f=j.results?.[0];if(!f)return null;return{name:f.name||"",admin1:f.admin1||"",country_code:f.country_code||""}}

async function token(env){if(!env.SPOTIFY_CLIENT_ID||!env.SPOTIFY_CLIENT_SECRET)throw new Error("SPOTIFY_SECRETS_MISSING");if(globalThis.__spTok&&Date.now()<globalThis.__spTok.exp)return globalThis.__spTok.token;const creds=btoa(`${env.SPOTIFY_CLIENT_ID}:${env.SPOTIFY_CLIENT_SECRET}`);const r=await fetch("https://accounts.spotify.com/api/token",{method:"POST",headers:{Authorization:`Basic ${creds}`,"Content-Type":"application/x-www-form-urlencoded"},body:"grant_type=client_credentials"});const t=await r.text();if(!r.ok)throw new Error(`SPOTIFY_TOKEN_FAILED ${r.status}: ${t}`);const d=JSON.parse(t);globalThis.__spTok={token:d.access_token,exp:Date.now()+(d.expires_in-60)*1000};return d.access_token}

function escName(n){return String(n||"").replace(/"/g,'\\"').trim()}
async function searchCandidates(env,terms,market,daysBack,limit,rand,banNames){
  const tok=await token(env);
  const nowY=new Date().getUTCFullYear();
  const since=new Date(Date.now()-daysBack*86400000);
  const yr=`${since.getUTCFullYear()}-${nowY}`;
  const quoted=terms.map(t=>`"${t}"`).join(" OR ");
  const bans=(banNames||[]).map(n=>n&&`NOT artist:"${escName(n)}"`).filter(Boolean).slice(0,6).join(" ");
  const core=`(${quoted}) year:${yr}`;
  const q=bans?`${core} ${bans}`:core;
  const offset=Math.floor(rand()*90);
  const p=new URLSearchParams({q,type:"track",market,limit:String(limit),offset:String(offset),include_external:"audio"});
  const r=await fetch(`https://api.spotify.com/v1/search?${p}`,{headers:{Authorization:`Bearer ${tok}`}});
  if(!r.ok)return[];
  const j=await r.json();
  const items=j.tracks?.items||[];
  return items.map(t=>({id:t.id,name:t.name,artists:t.artists,album_id:t.album?.id||null,album_release:parseDate(t.album?.release_date,t.album?.release_date_precision),popularity:t.popularity,preview_url:t.preview_url,url:`https://open.spotify.com/track/${t.id}`,app_uri:`spotify://track/${t.id}`}))
}
function parseDate(iso,prec){if(!iso)return null; if(prec==="day"||!prec)return new Date(iso+"T12:00:00Z"); if(prec==="month")return new Date(iso+"-15T12:00:00Z"); if(prec==="year")return new Date(iso+"-07-01T12:00:00Z"); return new Date(iso)}

async function newReleases(env,market,limit=30){
  const tok=await token(env);
  const r=await fetch(`https://api.spotify.com/v1/browse/new-releases?country=${market}&limit=${limit}`,{headers:{Authorization:`Bearer ${tok}`}});
  if(!r.ok)return[];
  const j=await r.json();
  const albums=(j.albums?.items||[]).slice(0,limit);
  if(!albums.length)return[];
  const ids=albums.map(a=>a.id).slice(0,20).join(",");
  const ra=await fetch(`https://api.spotify.com/v1/albums?ids=${ids}&market=${market}`,{headers:{Authorization:`Bearer ${tok}`}});
  if(!ra.ok)return[];
  const ja=await ra.json();
  const tr=[];
  for(const a of (ja.albums||[])){const rel=parseDate(a.release_date,a.release_date_precision);for(const t of (a.tracks?.items||[])){tr.push({id:t.id,name:t.name,artists:t.artists,album_id:a.id,album_release:rel,popularity:null})}}
  const slice=tr.slice(0,40);if(!slice.length)return[];
  const tid=slice.map(t=>t.id).join(",");
  const rt=await fetch(`https://api.spotify.com/v1/tracks?ids=${tid}`,{headers:{Authorization:`Bearer ${tok}`}});
  if(!rt.ok)return slice.map(t=>({...t,url:`https://open.spotify.com/track/${t.id}`,app_uri:`spotify://track/${t.id}`}));
  const jt=await rt.json();
  const map=new Map((jt.tracks||[]).map(x=>[x.id,x]));
  return slice.map(t=>{const f=map.get(t.id);return f?{...t,popularity:f.popularity,preview_url:f.preview_url,url:`https://open.spotify.com/track/${t.id}`,app_uri:`spotify://track/${t.id}`}:{...t,url:`https://open.spotify.com/track/${t.id}`,app_uri:`spotify://track/${t.id}`}})
}

async function enrichAF(env,tracks){
  const tok=await token(env);
  const ids=tracks.map(t=>t.id).slice(0,100).join(",");if(!ids)return tracks;
  const r=await fetch(`https://api.spotify.com/v1/audio-features?ids=${ids}`,{headers:{Authorization:`Bearer ${tok}`}});
  if(!r.ok)return tracks;
  const j=await r.json();
  const m=new Map((j.audio_features||[]).filter(Boolean).map(f=>[f.id,f]));
  return tracks.map(t=>({...t,af:m.get(t.id)||null}))
}

function score(r,t,m,recent){
  const now=Date.now();
  const rel=t.album_release?((now-t.album_release.getTime())/86400000):9999;
  const rec=Math.exp(-rel/90);
  const pop=typeof t.popularity==="number"?t.popularity:50;
  const popS=(pop>=30&&pop<=70)?1:(pop<30?0.85:0.6);
  let afS=0.6;
  if(t.af){const dE=Math.abs((t.af.energy??0.5)-m.energy);const dV=Math.abs((t.af.valence??0.5)-m.valence);const target=(m.energy+m.valence)/2;const dD=Math.abs((t.af.danceability??0.5)-target);afS=1-Math.min(1,(dE*0.45+dV*0.45+dD*0.35))}
  const aids=recent?.aids||new Set();const an=recent?.an||new Set();
  const names=(t.artists||[]).map(a=>(a.name||"").toLowerCase());
  const seenId=(t.artists||[]).some(a=>a?.id&&aids.has(a.id));
  const seenNm=names.some(n=>an.has(n));
  const pen=(seenId?0.18:0)+(seenNm?0.08:0);
  const ser=0.25+r()*0.25;
  const noise=(r()*2-1)*0.07;
  return rec*0.38+popS*0.18+afS*0.34+(ser*noise)-pen
}

function dedupByArtistAlbum(arr){const seenA=new Set();const seenAlb=new Set();const out=[];for(const t of arr){const aId=(t.artists?.[0]?.id)||null;const alb=t.album_id||null;if(aId&&seenA.has(aId))continue;if(alb&&seenAlb.has(alb))continue;if(aId)seenA.add(aId);if(alb)seenAlb.add(alb);out.push(t)}return out}

async function recommend(env,rand,mood,fam,terms,market,recent){
  const baseDays=120+Math.floor(rand()*240);
  const limit=30+Math.floor(rand()*20);
  const banNames=recent?Array.from(recent.an||[]).slice(0,6):[];
  let poolA=await searchCandidates(env,terms,market,baseDays,limit,rand,banNames);
  const newR=await newReleases(env,market,30);
  let pool=[...poolA,...newR];
  const ids=new Set();pool=pool.filter(t=>t.id&&!ids.has(t.id)&&(ids.add(t.id),true));
  pool=await enrichAF(env,pool);
  if(recent){const tids=recent.tids||new Set();const aids=recent.aids||new Set();const an=recent.an||new Set();let filtered=pool.filter(t=>{if(!t||!t.id)return false;if(tids.has(t.id))return false;const a=t.artists||[];if(a.some(x=>x&&x.id&&aids.has(x.id)))return false;if(a.some(x=>(x?.name||"").trim()&&an.has(x.name.toLowerCase())))return false;return true});pool=filtered.length>=5?filtered:pool.filter(t=>!tids.has(t.id))}
  pool=dedupByArtistAlbum(pool);
  if(pool.length<6){const fams=["chill","jazz","brightpop","groove","acoustic","soul"];const alt=fams.filter(x=>x!==fam)[Math.floor(rand()*(fams.length-1))];const seeds=FAMILY_SEEDS[alt]||[];const altTerms=seedTerms(seeds.length?seeds:terms);const more=await searchCandidates(env,altTerms,market,365,limit,rand,banNames);let extra=await enrichAF(env,more);const seen=new Set(pool.map(x=>x.id));extra=extra.filter(x=>!seen.has(x.id));pool=[...pool,...dedupByArtistAlbum(extra)]}
  if(pool.length<6&&market!=="US"){const moreUS=await searchCandidates(env,terms,"US",365,limit,rand,banNames);pool=[...pool,...dedupByArtistAlbum(await enrichAF(env,moreUS))]}
  if(!pool.length)return null;
  const scored=pool.map(t=>({t,s:score(rand,t,mood,recent)})).sort((a,b)=>b.s-a.s);
  const short=scored.slice(0,20+Math.floor(rand()*10));
  const pick=short[Math.floor(rand()*Math.min(short.length,16))]?.t||scored[0].t;
  return{id:pick.id,name:pick.name,artists:(pick.artists||[]).map(a=>a.name).join(", "),artist_ids:(pick.artists||[]).map(a=>a.id).filter(Boolean),url:pick.url,app_uri:pick.app_uri,preview_url:pick.preview_url}
}

function json(data,init={}){return new Response(JSON.stringify(data,null,2),{headers:{"content-type":"application/json; charset=utf-8","access-control-allow-origin":"*"},...init})}

function page(){
return new Response(`<!doctype html><html lang="es"><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>no24</title><style>:root{color-scheme:dark}html,body{height:100%}body{margin:0;background:#000;color:#fff;display:grid;place-items:center;font:clamp(18px,2.5vw,28px)/1.3 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif}.center{text-align:center}.step{font-weight:600;opacity:0;animation:fadeIn .4s ease forwards}.info{margin-top:12px;min-height:1.6em}.bubble{display:inline-block;opacity:0;transform:translateY(8px)}.fade{animation:infoFade 1.4s ease forwards}@keyframes fadeIn{from{opacity:0}to{opacity:1}}@keyframes infoFade{0%{opacity:0;transform:translateY(8px)}15%{opacity:1;transform:translateY(0)}80%{opacity:1}100%{opacity:0;transform:translateY(-6px)}}</style><div class="center"><div id="step" class="step">1. Analizando clima</div><div id="info" class="info"></div></div><script>(function(){var $=s=>document.querySelector(s);var step=$("#step"),info=$("#info");function setStep(t){step.textContent=t;step.classList.remove("step");void step.offsetWidth;step.classList.add("step")}function sleep(ms){return new Promise(r=>setTimeout(r,ms))}function showOnce(txt,d){return new Promise(function(res){info.innerHTML="";var s=document.createElement("span");s.className="bubble fade";s.textContent=txt;info.appendChild(s);setTimeout(res,d||1400)})}function localGet(k){try{return JSON.parse(localStorage.getItem(k)||"{}")}catch{return{}}}function localSet(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch{}}function prune(obj,ttl){var now=Date.now();var out={};for(var id in obj){if(obj[id]&&now-obj[id]<ttl)out[id]=obj[id]}return out}function banPayload(){var A=prune(localGet("n24ban_art")||{},21600000);var T=prune(localGet("n24ban_trk")||{},21600000);var N=prune(localGet("n24ban_names")||{},21600000);localSet("n24ban_art",A);localSet("n24ban_trk",T);localSet("n24ban_names",N);return{ban_art:Object.keys(A).slice(0,30).join(","),ban_trk:Object.keys(T).slice(0,30).join(","),ban_names:Object.keys(N).slice(0,30).join(",")}}function banRemember(track){try{var A=localGet("n24ban_art")||{};var T=localGet("n24ban_trk")||{};var N=localGet("n24ban_names")||{};var now=Date.now();(track.artist_ids||[]).forEach(id=>A[id]=now);if(track.id)T[track.id]=now;String(track.artists||"").split(",").map(s=>s.trim().toLowerCase()).filter(Boolean).forEach(n=>N[n]=now);localSet("n24ban_art",A);localSet("n24ban_trk",T);localSet("n24ban_names",N)}catch{}}function geoExact(){return new Promise(function(res){navigator.geolocation.getCurrentPosition(p=>res({ok:true,lat:p.coords.latitude,lon:p.coords.longitude,approx:false}),()=>res({ok:false}),{enableHighAccuracy:true,timeout:9000,maximumAge:0})})}async function geoApprox(){try{var u=new URL(location.href);u.pathname="/api/where";var r=await fetch(u.toString());var j=await r.json();if(j&&j.ok&&typeof j.lat==="number"&&typeof j.lon==="number")return{ok:true,lat:j.lat,lon:j.lon,approx:true,city:j.city,tz:j.timezone}}catch{}return{ok:false}}(async function run(){try{setStep("1. Analizando clima");var exact=await geoExact();var pos=exact.ok?exact:await geoApprox();if(!pos.ok)throw new Error("geo");var uMeta=new URL(location.href);uMeta.pathname="/api/meta";uMeta.search=new URLSearchParams({lat:pos.lat,lon:pos.lon}).toString();var rMeta=await fetch(uMeta.toString());var m=await rMeta.json();var loc=m&&m.city?m.city:(pos.city||"Tu ubicación");var tC=m&&typeof m.temperature==="number"?Math.round(m.temperature)+"°C":"";await showOnce("Ciudad: "+loc+(pos.approx?" (aprox.)":""),1400);await showOnce("Temperatura: "+tC,1400);setStep("2. Consiguiendo una canción");var u=new URL(location.href);u.pathname="/api/track";var s=new Uint32Array(1);crypto.getRandomValues(s);var salt=(s[0]||Date.now())>>>0;if(salt===0)salt=1;var bans=banPayload();u.search=new URLSearchParams({lat:pos.lat,lon:pos.lon,spice:"0.35",salt:String(salt),ban_art:bans.ban_art,ban_trk:bans.ban_trk,ban_names:bans.ban_names}).toString();var r=await fetch(u.toString());var j=await r.json();if(!r.ok||!j||!j.track||!j.track.id)throw new Error(j&&j.error||"Sin canción");banRemember(j.track);for(var n=3;n>=1;n--){setStep("3. Abriendo Spotify en "+n);await sleep(1000)}var app=j.track.app_uri||("spotify://track/"+j.track.id);var web=j.track.url||("https://open.spotify.com/track/"+j.track.id);var did=false;document.addEventListener("visibilitychange",function(){if(document.hidden)did=true},{once:true});setTimeout(function(){if(!did)location.replace(web)},1800);location.href=app}catch(e){step.textContent="No pude obtener ubicación o canción. Revisa permisos y recarga.";info.textContent=""}})()})();</script></html>`,{headers:{"content-type":"text/html; charset=utf-8"}})
}

function parseCsvIds(s){return String(s||"").split(",").map(x=>x.trim()).filter(x=>/^[A-Za-z0-9]{10,64}$/.test(x))}
function parseCsvNames(s){return String(s||"").split(",").map(x=>x.trim().toLowerCase()).filter(Boolean)}

export default{
async fetch(req,env){
  const url=new URL(req.url);const {pathname,searchParams}=url;
  const cfCountry=req.cf?.country;const market=/^[A-Z]{2}$/.test(cfCountry||"")?cfCountry:"US";
  if(pathname==="/")return page();

  if(pathname==="/api/meta"){
    try{const lat=Number(searchParams.get("lat"));const lon=Number(searchParams.get("lon"));if(!Number.isFinite(lat)||!Number.isFinite(lon))return json({error:"lat/lon requeridos"},{status:400});const cw=await fetchWeather(env,lat,lon);const rg=await reverseGeocode(env,lat,lon);const city=rg?(rg.name+(rg.admin1?(", "+rg.admin1):"")+(rg.country_code?(" • "+rg.country_code):"")):null;const tz=req.cf?.timezone||"UTC";return json({city,temperature:cw.temperature,is_day:cw.is_day,weathercode:cw.weathercode,time:cw.time,timezone:tz})}catch(e){return json({error:String(e?.message||e)},{status:500})}
  }

  if(pathname==="/api/where"){
    try{const lat=Number(req.cf?.latitude);const lon=Number(req.cf?.longitude);const city=req.cf?.city||null;const tz=req.cf?.timezone||"UTC";if(!Number.isFinite(lat)||!Number.isFinite(lon))return json({ok:false},{status:200});return json({ok:true,lat,lon,city,timezone:tz})}catch{return json({ok:false},{status:200})}
  }

  if(pathname==="/api/track"){
    const debug=searchParams.get("debug")==="1";
    try{
      const lat=Number(searchParams.get("lat"));const lon=Number(searchParams.get("lon"));const spiceParam=Number(searchParams.get("spice"));
      if(!Number.isFinite(lat)||!Number.isFinite(lon))return json({error:"lat/lon requeridos"},{status:400});
      const cw=await fetchWeather(env,lat,lon);
      const feat=featFromWx(cw);
      const mood0=inferMood(feat);
      const doy=dayOfYear(feat.timeISO);
      const baseSeed=(Math.floor(lat*1e4)^Math.floor(lon*1e4)^(feat.hour<<8)^(doy<<16))|0;
      const saltParam=Number(searchParams.get("salt"));
      let salt32;if(Number.isFinite(saltParam))salt32=(saltParam|0)||((Date.now()&0xffffffff)|1);else{const a=new Uint32Array(1);crypto.getRandomValues(a);salt32=(a[0]|0)||((Date.now()&0xffffffff)|1)}
      const rand=rngSeed(baseSeed^salt32);

      let mood={...mood0};
      if(feat.bucket===3||feat.bucket===2){mood.energy=clamp(mood.energy-0.1,0,1);mood.valence=clamp(mood.valence-0.05,0,1)}
      if(feat.hour>=6&&feat.hour<12)mood.valence=clamp(mood.valence+0.1,0,1);
      if(feat.hour>=18&&feat.hour<24)mood.energy=clamp(mood.energy-0.05,0,1);
      const spice=Number.isFinite(spiceParam)?clamp(spiceParam,0,1):0.35;
      mood.energy=clamp(mood.energy+(rand()*2-1)*0.1*spice,0,1);
      mood.valence=clamp(mood.valence+(rand()*2-1)*0.1*spice,0,1);

      const ctx={hour:feat.hour,weekend:[5,6,0].includes(new Date(feat.timeISO).getUTCDay()),lat,lon};
      const weights=ctxWeights(ctx,feat,mood);
      const fam=pickWeighted(rand,Object.keys(weights).map(k=>({k,w:Math.max(0.1,weights[k])})));
      const fams=["chill","jazz","brightpop","groove","acoustic","soul"];
      const neighbor=fams.filter(x=>x!==fam)[Math.floor(rand()*(fams.length-1))];
      const terms=seedTerms((FAMILY_SEEDS[fam]||[]).map(normGenre).filter(g=>!banned.has(g)).length?(FAMILY_SEEDS[fam]||[]):buildTerms(rand,fam,neighbor));

      const key=recKey(lat,lon);
      const memR=recGet(key);
      const cok=readCookie(req);
      const recent={tids:new Set([...memR.tids,...cok.tids]),aids:new Set([...memR.aids,...cok.aids]),an:new Set([...memR.an,...cok.an])};

      const ban_art=parseCsvIds(searchParams.get("ban_art"));
      const ban_trk=parseCsvIds(searchParams.get("ban_trk"));
      const ban_names=parseCsvNames(searchParams.get("ban_names"));
      ban_art.forEach(x=>recent.aids.add(x));
      ban_trk.forEach(x=>recent.tids.add(x));
      ban_names.forEach(x=>recent.an.add(x));

      const track=await recommend(env,rand,mood,fam,terms,market,recent);
      if(!track)return json({error:"Sin recomendaciones",weather:cw,features:mood,family:fam,terms:terms},{status:502});

      recPush(key,track);
      const next={tids:new Set(recent.tids),aids:new Set(recent.aids),an:new Set(recent.an)};
      next.tids.add(track.id);(track.artist_ids||[]).forEach(id=>next.aids.add(id));String(track.artists||"").split(",").map(s=>s.trim().toLowerCase()).filter(Boolean).forEach(n=>next.an.add(n));
      const cookie=writeCookie(next);

      const buckets=["despejado","nublado","niebla","lluvia","nieve","tormenta"];
      const summary=buckets[feat.bucket]+" • "+(cw.is_day?"día":"noche");
      const payload={weather:{...cw,summary},context:{family:fam,terms,market,spice,salt:salt32},features:mood,track:{id:track.id,name:track.name,artists:track.artists,artist_ids:track.artist_ids,url:track.url,app_uri:track.app_uri,preview_url:track.preview_url}};
      const resp=json(debug?{debug:true,...payload}:payload);
      resp.headers.set("Set-Cookie",cookie);
      return resp;
    }catch(e){return json({error:String(e?.message||e)},{status:500})}
  }

  if(pathname==="/go"){
    try{
      const lat=Number(searchParams.get("lat"));const lon=Number(searchParams.get("lon"));const spiceParam=Number(searchParams.get("spice"));
      const cw=await fetchWeather(env,lat,lon);const feat=featFromWx(cw);const mood0=inferMood(feat);
      const doy=dayOfYear(feat.timeISO);const baseSeed=(Math.floor(lat*1e4)^Math.floor(lon*1e4)^(feat.hour<<8)^(doy<<16))|0;
      const saltParam=Number(searchParams.get("salt"));let salt32;if(Number.isFinite(saltParam))salt32=(saltParam|0)||((Date.now()&0xffffffff)|1);else{const a=new Uint32Array(1);crypto.getRandomValues(a);salt32=(a[0]|0)||((Date.now()&0xffffffff)|1)};const rand=rngSeed(baseSeed^salt32);
      let mood={...mood0};if(feat.bucket===3||feat.bucket===2){mood.energy=clamp(mood.energy-0.1,0,1);mood.valence=clamp(mood.valence-0.05,0,1)};if(feat.hour>=6&&feat.hour<12)mood.valence=clamp(mood.valence+0.1,0,1);if(feat.hour>=18&&feat.hour<24)mood.energy=clamp(mood.energy-0.05,0,1);const spice=Number.isFinite(spiceParam)?clamp(spiceParam,0,1):0.35;mood.energy=clamp(mood.energy+(rand()*2-1)*0.1*spice,0,1);mood.valence=clamp(mood.valence+(rand()*2-1)*0.1*spice,0,1);
      const ctx={hour:feat.hour,weekend:[5,6,0].includes(new Date(feat.timeISO).getUTCDay()),lat,lon};const weights=ctxWeights(ctx,feat,mood);const fams=Object.keys(weights);const fam=pickWeighted(rand,fams.map(k=>({k,w:Math.max(0.1,weights[k])})));const neighbor=["chill","jazz","brightpop","groove","acoustic","soul"].filter(x=>x!==fam)[Math.floor(rand()*5)];const terms=seedTerms((FAMILY_SEEDS[fam]||[]).map(normGenre).filter(g=>!banned.has(g)).length?(FAMILY_SEEDS[fam]||[]):buildTerms(rand,fam,neighbor));
      const key=recKey(lat,lon);const memR=recGet(key);const cok=readCookie(req);const recent={tids:new Set([...memR.tids,...cok.tids]),aids:new Set([...memR.aids,...cok.aids]),an:new Set([...memR.an,...cok.an])};
      const ban_art=parseCsvIds(searchParams.get("ban_art"));const ban_trk=parseCsvIds(searchParams.get("ban_trk"));const ban_names=parseCsvNames(searchParams.get("ban_names"));ban_art.forEach(x=>recent.aids.add(x));ban_trk.forEach(x=>recent.tids.add(x));ban_names.forEach(x=>recent.an.add(x));
      const track=await recommend(env,rand,mood,fam,terms,/^[A-Z]{2}$/.test(req.cf?.country||"")?req.cf.country:"US",recent);
      if(track){recPush(key,track);const next={tids:new Set(recent.tids),aids:new Set(recent.aids),an:new Set(recent.an)};next.tids.add(track.id);(track.artist_ids||[]).forEach(id=>next.aids.add(id));String(track.artists||"").split(",").map(s=>s.trim().toLowerCase()).filter(Boolean).forEach(n=>next.an.add(n));const cookie=writeCookie(next);const resp=Response.redirect(`spotify://track/${track.id}`,302);resp.headers.set("Set-Cookie",cookie);return resp}
      return Response.redirect("https://open.spotify.com/",302)
    }catch{return Response.redirect("https://open.spotify.com/",302)}
  }

  return new Response("Not found",{status:404})
}}