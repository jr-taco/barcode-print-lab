import {decodeWatermark,readEAN13} from './watermark-osd.mjs';
const el=id=>document.getElementById(id), canvas=el('preview'), video=el('video');
const w=452,h=336;
let source, points=[], url, running=false, engine;
function clear(){
 for(const id of ['public','hidden'])el(id).textContent='等待读取';
 for(const id of ['public-detail','hidden-detail','status'])el(id).textContent='';
}
function display(image,width,height,aligned=false){
 const scale=Math.min(1,1500/Math.max(width,height));
 canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);
 canvas.hidden=false;el('reset').hidden=false;
 const ctx=canvas.getContext('2d',{willReadFrequently:true});
 ctx.drawImage(image,0,0,canvas.width,canvas.height);
 source=ctx.getImageData(0,0,canvas.width,canvas.height);
 points=aligned?[[0,0],[canvas.width-1,0],[canvas.width-1,canvas.height-1],[0,canvas.height-1]]:[];
 el('read').disabled=points.length!==4;
 clear();
}
function rectified(){
 const rows=[];
 [[0,0],[1,0],[1,1],[0,1]].forEach(([u,v],i)=>{
   const [x,y]=points[i];
   rows.push([u,v,1,0,0,0,-u*x,-v*x,x],[0,0,0,u,v,1,-u*y,-v*y,y]);
 });
 for(let c=0;c<8;c++){
   let p=c;for(let j=c+1;j<8;j++)if(Math.abs(rows[j][c])>Math.abs(rows[p][c]))p=j;
   if(Math.abs(rows[p][c])<1e-9)throw Error('四角无效，请重选。');
   [rows[c],rows[p]]=[rows[p],rows[c]];
   const s=rows[c][c];rows[c]=rows[c].map(x=>x/s);
   for(let j=0;j<8;j++)if(j!==c){const f=rows[j][c];rows[j]=rows[j].map((x,k)=>x-f*rows[c][k]);}
 }
 const [a,b,c,d,e,f,g,i]=rows.map(x=>x[8]), pixels=new Float32Array(w*h);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const u=(x+.5)/w,v=(y+.5)/h,z=g*u+i*v+1;
   const sx=Math.max(0,Math.min(source.width-1,(a*u+b*v+c)/z));
   const sy=Math.max(0,Math.min(source.height-1,(d*u+e*v+f)/z));
   const pos=4*(Math.round(sy)*source.width+Math.round(sx));
   const lum=.299*source.data[pos]+.587*source.data[pos+1]+.114*source.data[pos+2];
   pixels[y*w+x]=lum;
 }
 return {pixels};
}
canvas.onclick=e=>{
 if(!source||running||points.length===4)return;
 const r=canvas.getBoundingClientRect();
 points.push([(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]);
 const ctx=canvas.getContext('2d');ctx.fillStyle='#d23420';
 const [x,y]=points.at(-1);ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);ctx.fill();
 el('read').disabled=points.length!==4;
};
el('reset').onclick=()=>{if(running)return;points=[];canvas.getContext('2d').putImageData(source,0,0);el('read').disabled=true;clear();};
el('file').onchange=async()=>{
 if(running)return;
 const file=el('file').files[0];if(!file)return;
 video.hidden=true;el('capture').hidden=true;
 source=undefined;canvas.hidden=true;el('reset').hidden=true;el('read').disabled=true;clear();
 if(url)URL.revokeObjectURL(url);
 url=URL.createObjectURL(file);
 if(file.type.startsWith('video/')||/\.(mp4|mov|m4v|webm)$/i.test(file.name)){
   video.src=url;video.hidden=false;el('capture').hidden=false;
 }else{
   try{const image=await createImageBitmap(file);display(image,image.width,image.height);image.close();}
   catch{el('status').textContent='无法读取此照片。';}
 }
};
el('capture').onclick=()=>{if(!running&&video.videoWidth)display(video,video.videoWidth,video.videoHeight);};
el('example').onclick=async()=>{
 if(running)return;
 try{const r=await fetch('watermark-sample.png');if(!r.ok)throw Error();
   const im=await createImageBitmap(await r.blob()); display(im,im.width,im.height,true);im.close();await read();}
 catch{el('status').textContent='无法读取数字样张。';}
};
async function load(){
 if(!globalThis.ort)throw Error('浏览器推理组件加载失败。');
 ort.env.wasm.numThreads=1;
 ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
 const [session,reference]=await Promise.all([
   ort.InferenceSession.create('watermark-reader.onnx',{executionProviders:['wasm']}),
   fetch('watermark-reference.json').then(r=>{if(!r.ok)throw Error('样张参考文件读取失败。');return r.json();})
 ]);
 return {session,reference};
}
async function read(){
 if(running||points.length!==4)return;
 running=true;el('read').disabled=true;clear();el('status').textContent='正在解码…';
 try{
   const {session,reference}=await (engine??=load().catch(e=>{engine=undefined;throw e;}));
   const {pixels}=rectified();
   const values=readEAN13(pixels,w,h);
   el('public').textContent=!values.length?'未读出':values.length===1&&values[0]===reference.ean?'正确':'不匹配';
   el('public-detail').textContent='预期：'+reference.ean+'；实际：'+(values.join('、')||'—');
   const output=await session.run({luminance:new ort.Tensor('float32',pixels,[1,1,h,w])});
   const payload=decodeWatermark(output.logits.data,reference.generator);
   el('hidden').textContent=payload===reference.payload?'正确':'不匹配';
   el('hidden-detail').textContent='读出载荷：'+payload+'（与此样张的参考值比对）';
   el('status').textContent='核对完成';
 }catch(e){el('hidden').textContent='未验证';el('status').textContent=e.message||'读取失败。';}
 finally{running=false;el('read').disabled=false;}
}
el('read').onclick=read;
