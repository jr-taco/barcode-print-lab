// Order-3 reliability-based decoding over the bundled BCH generator.
export function decodeWatermark(scores, generator) {
  const k = 56, n = 91;
  if (scores.length < n || Array.from(scores).some(x => !Number.isFinite(x))) throw Error('无效模型输出');
  const perm = Array.from({length:n}, (_,i)=>i).sort((a,b)=>Math.abs(scores[b])-Math.abs(scores[a]));
  const g = generator.map(row=>Uint8Array.from(perm,i=>row[i]));
  const y = Uint8Array.from(perm,i=>+(scores[i]>0));
  const rel = Float64Array.from(perm,i=>Math.abs(scores[i]));
  const piv=[];
  for(let col=0;col<n&&piv.length<k;col++){
    const j=piv.length, r=g.findIndex((row,i)=>i>=j&&row[col]);
    if(r<0)continue;
    [g[j],g[r]]=[g[r],g[j]];
    for(let i=0;i<k;i++)if(i!==j&&g[i][col])for(let b=0;b<n;b++)g[i][b]^=g[j][b];
    piv.push(col);
  }
  if(piv.length!==k)throw Error('无效纠错矩阵');
  const base=new Uint8Array(n);
  for(let j=0;j<k;j++)if(y[piv[j]])for(let b=0;b<n;b++)base[b]^=g[j][b];
  const err=base.map((v,i)=>v^y[i]);
  const zero=new Uint8Array(n);
  let best=Infinity, flips=[];
  function consider(a=-1,b=-1,c=-1){
    const x=g[a]||zero,z=g[b]||zero,w=g[c]||zero;
    let metric=0;
    for(let i=0;i<n;i++) if(err[i]^x[i]^z[i]^w[i])metric+=rel[i];
    if(metric<best){best=metric;flips=[a,b,c].filter(i=>i>=0);}
  }
  consider();
  for(let a=0;a<k;a++)consider(a);
  for(let a=0;a<k;a++)for(let b=a+1;b<k;b++)consider(a,b);
  for(let a=0;a<k;a++)for(let b=a+1;b<k;b++)for(let c=b+1;c<k;c++)consider(a,b,c);
  const result=new Uint8Array(k);
  for(let i=0;i<n;i++)if(perm[i]<k){
    let bit=base[i];for(const f of flips)bit^=g[f][i];result[perm[i]]=bit;
  }
  let hex='';
  for(let i=0;i<k;i+=8){
    let byte=0;for(let j=0;j<8;j++)byte=2*byte+result[i+j];
    hex+=byte.toString(16).padStart(2,'0');
  }
  return hex;
}

// EAN-13 verification on a rectified full-symbol crop.
export function readEAN13(pixels,w,h){
  const left=['0001101','0011001','0010011','0111101','0100011','0110001','0101111','0111011','0110111','0001011'];
  const invert=s=>s.replace(/[01]/g,b=>b==='0'?'1':'0');
  const right=left.map(invert);
  const even=right.map(s=>[...s].reverse().join(''));
  const parity=['LLLLLL','LLGLGG','LLGGLG','LLGGGL','LGLLGG','LGGLLG','LGGGLL','LGLGLG','LGLGGL','LGGLGL'];
  const values=new Set();
  for(const frac of [.25,.35,.45,.55,.65]){
    const row=pixels.subarray(Math.round(h*frac)*w,Math.round(h*frac)*w+w);
    const lo=Math.min(...row), hi=Math.max(...row);if(hi-lo<65)continue;
    const threshold=(lo+hi)/2;
    const start=row.findIndex(v=>v<threshold), end=row.findLastIndex(v=>v<threshold);
    if(end-start<95)continue;
    const units=(end+1-start)/95;
    let bits='';
    for(let i=0;i<95;i++)bits+=row[Math.floor(start+(i+.5)*units)]<threshold?'1':'0';
    if(!bits.startsWith('101')||bits.slice(45,50)!=='01010'||!bits.endsWith('101'))continue;
    let digits='', sides='';
    for(let i=0;i<6;i++){
      const word=bits.slice(3+i*7,10+i*7), l=left.indexOf(word), g=even.indexOf(word);
      if(l<0&&g<0)break;
      digits+=l>=0?l:g;sides+=l>=0?'L':'G';
    }
    const first=parity.indexOf(sides);if(first<0)continue;
    digits=first+digits;
    for(let i=0;i<6;i++){
      const d=right.indexOf(bits.slice(50+i*7,57+i*7));if(d<0)break;
      digits+=d;
    }
    if(digits.length===13&&[...digits].reduce((a,c,i)=>a+Number(c)*(i%2?3:1),0)%10===0)values.add(digits);
  }
  return [...values];
}
