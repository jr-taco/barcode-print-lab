import {alignRough} from './align.mjs';
import {decodeWatermark} from '../watermark-osd.mjs';

let manifest, generator, runtime, loadedSession;
function json(url){
 return fetch(new URL(url,import.meta.url)).then(r=>{if(!r.ok)throw Error('读取配置失败');return r.json();});
}
function runtimeReady(){
 if(runtime)return runtime;
 runtime=new Promise((resolve,reject)=>{
   if(globalThis.ort){resolve(globalThis.ort);return;}
   const script=document.createElement('script');
   script.src='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js';
   script.onload=()=>globalThis.ort?resolve(globalThis.ort):reject(Error('模型运行组件不可用'));
   script.onerror=()=>{script.remove();reject(Error('模型运行组件加载失败'));};
   document.head.append(script);
 }).catch(e=>{runtime=undefined;throw e;});
 return runtime;
}
export async function configureRough(uid){
 manifest??=json('./samples.json').catch(e=>{manifest=undefined;throw e;});
 const row=(await manifest).find(r=>r.uid===uid);
 if(!row)throw Error('样张的水印配置不存在');
 if(row.control)return {collect(){},async result(){return {status:'control',frames:0};}};
 const frames=[];
 return {
   collect(image,hits){
     const matches=hits.filter(b=>b.text===row.value&&b.position);
     if(matches.length!==1)return;
     const item=alignRough(image,matches[0].position,row);
     if(!item)return;
     frames.push(item);frames.sort((a,b)=>a.residual-b.residual);
     if(frames.length>5)frames.pop();
   },
   async result(){
     if(!frames.length)return {status:'no_read',frames:0};
     const ort=await runtimeReady();
     ort.env.wasm.numThreads=1;
     ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
     if(loadedSession?.model!==row.model){
       if(loadedSession)await loadedSession.session.release();
       loadedSession=undefined;
       const session=await ort.InferenceSession.create(new URL(row.model,import.meta.url).href,{executionProviders:['wasm']});
       loadedSession={model:row.model,session};
     }
     const scores=new Float64Array(100);
     for(const frame of frames){
       const out=await loadedSession.session.run({luminance:new ort.Tensor('float32',frame.pixels,[1,1,336,452])});
       for(let i=0;i<100;i++)scores[i]+=out.logits.data[i];
     }
     generator??=json('../watermark-reference.json').then(r=>r.generator).catch(e=>{generator=undefined;throw e;});
     const payload=decodeWatermark(scores,await generator);
     return {status:payload===row.reference?'correct':'mismatch',frames:frames.length,payload};
   }
 };
}
