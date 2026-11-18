// Align from public bar positions. The hidden payload is never used in registration.
function fit(pairs) {
  const m=Array.from({length:8},()=>new Float64Array(9));
  for(const [[x,y],[u,v]] of pairs){
    for(const r of [[x,y,1,0,0,0,-u*x,-u*y,u],[0,0,0,x,y,1,-v*x,-v*y,v]]){
      for(let i=0;i<8;i++)for(let j=0;j<9;j++)m[i][j]+=r[i]*r[j];
    }
  }
  for(let i=0;i<8;i++){
    let p=i;
    for(let j=i+1;j<8;j++)if(Math.abs(m[j][i])>Math.abs(m[p][i]))p=j;
    if(Math.abs(m[p][i])<1e-7)throw Error('定位点不足');
    [m[i],m[p]]=[m[p],m[i]];
    const d=m[i][i];for(let j=i;j<9;j++)m[i][j]/=d;
    for(let k=0;k<8;k++)if(k!==i){const a=m[k][i];for(let j=i;j<9;j++)m[k][j]-=a*m[i][j];}
  }
  return m.map(row=>row[8]);
}
function project(m,x,y){
  const z=m[6]*x+m[7]*y+1;
  return [(m[0]*x+m[1]*y+m[2])/z,(m[3]*x+m[4]*y+m[5])/z];
}
function grayscale(data){
  const gray=new Float32Array(data.length/4);
  for(let i=0;i<gray.length;i++)gray[i]=.299*data[i*4]+.587*data[i*4+1]+.114*data[i*4+2];
  return gray;
}
function sample(a,w,h,x,y){
  const i=Math.floor(x),j=Math.floor(y);
  if(i<0||j<0||i+1>=w||j+1>=h)return 255;
  const u=x-i,v=y-j,p=j*w+i;
  return (1-v)*((1-u)*a[p]+u*a[p+1])+v*((1-u)*a[p+w]+u*a[p+w+1]);
}
function otsu(im){
  const hist=new Uint32Array(256);
  for(const g of im)hist[Math.max(0,Math.min(255,Math.round(g)))]++;
  let sum=0;for(let i=0;i<256;i++)sum+=i*hist[i];
  let best=-1,chosen=128,n=0,first=0;
  for(let i=0;i<255;i++){
    n+=hist[i];first+=i*hist[i];if(!n||n===im.length)continue;
    const d=first/n-(sum-first)/(im.length-n),v=n*(im.length-n)*d*d;
    if(v>best){best=v;chosen=i;}
  }
  return chosen;
}
function findBars(im,w,h,threshold,off,entry){
  const visited=new Uint8Array(im.length),queue=new Int32Array(im.length),out=[];
  for(let p=0;p<im.length;p++){
    if(visited[p]||im[p]>threshold)continue;
    visited[p]=1;queue[0]=p;
    let size=1,xmin=w,xmax=0,ymin=h,ymax=0;
    for(let at=0;at<size;at++){
      const q=queue[at],x=q%w,y=(q-x)/w;
      xmin=Math.min(x,xmin);xmax=Math.max(x,xmax);ymin=Math.min(y,ymin);ymax=Math.max(y,ymax);
      for(const v of [x>0?q-1:-1,x<w-1?q+1:-1,y>0?q-w:-1,y<h-1?q+w:-1]){
        if(v>=0&&!visited[v]&&im[v]<=threshold){visited[v]=1;queue[size++]=v;}
      }
    }
    if(ymax-ymin<145||ymax-ymin>385||xmax-xmin>48||xmin-off<entry.left-28||xmax-off>entry.right+28)continue;
    const corners=[[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]];
    const best=corners.map(([x,y])=>{
      let dist=Infinity,at=0;
      for(let i=0;i<size;i++){
        const q=queue[i], px=q%w, py=(q-px)/w, dx=px-x,dy=py-y,d=dx*dx+dy*dy;
        if(d<dist){dist=d;at=i;}
      }
      const q=queue[at], px=q%w;
      return [px-off,(q-px)/w-off];
    });
    out.push({center:(xmin+xmax)/2-off,points:best});
  }
  return out.sort((a,b)=>a.center-b.center);
}
export function alignRough(image,position,entry){
  const w=452,h=336,offset=48,cw=w+2*offset,ch=h+2*offset;
  const {width:sw,height:sh}=image,gray=grayscale(image.data);
  const actual=['topLeft','topRight','bottomRight','bottomLeft'].map(k=>[position[k].x,position[k].y]);
  const [l,r,t,b]=[entry.left,entry.right-1,entry.top,entry.bottom-1];
  const initial=fit([[[l,t],actual[0]],[[r,t],actual[1]],[[r,b],actual[2]],[[l,b],actual[3]]]);
  const coarse=new Float32Array(cw*ch);
  for(let y=0;y<ch;y++)for(let x=0;x<cw;x++){
    const [sx,sy]=project(initial,x-offset,y-offset);
    coarse[y*cw+x]=sample(gray,sw,sh,sx,sy);
  }
  const bars=findBars(coarse,cw,ch,otsu(coarse)+4,offset,entry);
  const runs=[];
  for(let x=0;x<entry.pattern.length;x++){
    if(!entry.pattern[x])continue;
    const start=x;while(entry.pattern[x+1])x++;
    runs.push([start,x+1]);
  }
  if(bars.length!==runs.length)return null;
  const pairs=[];
  for(let i=0;i<runs.length;i++){
    const [a,z]=runs[i],xs=l+4*a,xe=l+4*z-1;
    const isGuard=entry.format==='EAN-13'&&(a<3||(a>=45&&a<50)||a>=92)||
                  entry.format==='EAN-8'&&(a<3||(a>=31&&a<36)||a>=64);
    const bottom=b+(isGuard?20:0);
    [[xs,t],[xe,t],[xe,bottom],[xs,bottom]].forEach((p,j)=>pairs.push([p,bars[i].points[j]]));
  }
  let refine=fit(pairs),eligible=pairs;
  for(const limit of [8,4,2.5]){
    const filtered=eligible.filter(([p,q])=>{
      const [x,y]=project(refine,...p);return Math.hypot(x-q[0],y-q[1])<limit;
    });
    if(filtered.length<pairs.length*.60)break;
    eligible=filtered;refine=fit(eligible);
  }
  const residual=eligible.reduce((sum,[p,q])=>{
    const [x,y]=project(refine,...p);return sum+Math.hypot(x-q[0],y-q[1]);
  },0)/eligible.length;
  if(residual>5)return null;
  const pixels=new Float32Array(w*h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const [cx,cy]=project(refine,x,y),[sx,sy]=project(initial,cx,cy);
    if(x>=l&&x<=r&&y>=t&&y<=b&&(sx<1||sy<1||sx>=sw-1||sy>=sh-1))return null;
    pixels[y*w+x]=sample(gray,sw,sh,sx,sy);
  }
  return {pixels,residual};
}
