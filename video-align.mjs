// Locate the bundled EAN-13 from its 95-module pattern on successive scanlines.
export function locateSample(image, pattern) {
  const {data,width,height}=image;
  const step=Math.max(1,Math.floor(height/400));
  const rows=[];
  let run=[];
  const row=new Float32Array(width);
  for(let y=0;y<height;y+=step){
    let low=255,high=0;
    for(let x=0;x<width;x++){
      const p=4*(y*width+x);
      const v=.299*data[p]+.587*data[p+1]+.114*data[p+2];
      row[x]=v;low=Math.min(low,v);high=Math.max(high,v);
    }
    let hit;
    if(high-low>65){
      const mid=(high+low)/2;
      const l=row.findIndex(v=>v<mid), r=row.findLastIndex(v=>v<mid)+1;
      const m=(r-l)/95;
      if(m>=1.2){
        let errors=0;
        for(let j=0;j<95;j++)errors+=+(+(row[Math.floor(l+(j+.5)*m)]<mid)!==pattern[j]);
        if(errors<=2)hit={y,l,r};
      }
    }
    if(hit&&(!run.length||y-run.at(-1).y<=3*step)){
      run.push(hit);
    }else if(run.length&&y-run.at(-1).y>2*step){
      rows.push(run);run=hit?[hit]:[];
    }else if(hit)run.push(hit);
  }
  rows.push(run);
  const best=rows.sort((a,b)=>b.length-a.length)[0];
  if(best.length<14)return null;
  const first=best[0],last=best.at(-1), span=last.y-first.y+step;
  const band=Math.max(2,Math.floor(best.length/8));
  const edge=(items,key)=>items.map(p=>p[key]).sort((a,b)=>a-b)[Math.floor(items.length/2)];
  const l0=edge(best.slice(0,band),'l'),r0=edge(best.slice(0,band),'r');
  const l1=edge(best.slice(-band),'l'),r1=edge(best.slice(-band),'r');
  if(span/(r0-l0)<.44||span/(r0-l0)>1.2)return null;
  const x0=(r0-l0)/380,x1=(r1-l1)/380,sy=span/280;
  return [[l0-44*x0,first.y-8*sy],[r0+28*x0,first.y-8*sy],
          [r1+28*x1,last.y+step+48*sy],[l1-44*x1,last.y+step+48*sy]];
}
