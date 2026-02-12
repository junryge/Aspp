import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

const SCALE = 0.5;
const R = (x) => (x - 480) * SCALE;
const RZ = (z) => (820 - z) * SCALE;

const buildings = [
  { name:"M14A/B", x:150, z:280, w:210, d:160, h:96, color:0xecc94b, type:"DRAM 생산동",
    detail:"M14A/B FAB동\n1a/1b nm DRAM 생산\n최신 공정 라인\n8층(옥상)",
    specs:{floors:"8F",process:"1a/1b nm",size:"210m×160m"},
    processFlow:["포토리소","식각","증착","CMP","이온주입"],
    chimneys:{count:1,height:50,radius:8}, fabNames:["M14A","M14B"]},
  { name:"M10A", x:160, z:520, w:200, d:120, h:65, color:0x63b3ed, type:"생산동",
    detail:"M10A FAB동\nDRAM 생산", specs:{floors:"5F",process:"DRAM"},
    processFlow:["웨이퍼투입","생산공정","검사","이송"], fabNames:["M10A"]},
  { name:"M10B/R3", x:500, z:300, w:140, d:100, h:60, color:0x63b3ed, type:"생산동",
    detail:"M10B/R3 FAB동\nDRAM 생산", specs:{floors:"5F"}, fabNames:["M10B"]},
  { name:"M10C", x:700, z:200, w:160, d:100, h:60, color:0x63b3ed, type:"생산동",
    detail:"M10C FAB동\nDRAM 생산", specs:{floors:"5F"}, fabNames:["M10C"]},
  { name:"M16A/B", x:530, z:520, w:230, d:120, h:132, color:0x48bb78, type:"DRAM 생산동",
    detail:"M16A/B FAB동\nAMHS 물류 시스템\n11층(옥상)",
    specs:{floors:"11F",logistics:"AMHS",size:"230m×120m"},
    processFlow:["DRAM양산","AMHS자동물류","클린룸","품질검증"],
    chimneys:{count:1,height:60,radius:9}, fabNames:["M16A","M16B"]},
  { name:"DRAM_WT", x:420, z:730, w:150, d:70, h:45, color:0xe07098, type:"웨이퍼 테스트",
    detail:"DRAM 웨이퍼 테스트동", specs:{floors:"4F",role:"검사"},
    processFlow:["프로브테스트","전기특성","수율판정","불량마킹"]},
  { name:"P&T1", x:420, z:140, w:110, d:80, h:35, color:0xc4956a, type:"패키지·테스트",
    detail:"P&T1 패키지 및 테스트동", specs:{floors:"3F"},
    processFlow:["다이싱","다이어태치","와이어본딩","몰딩","최종테스트"]},
  { name:"P&T4", x:830, z:320, w:140, d:120, h:96, color:0xc4956a, type:"패키지·테스트",
    detail:"P&T4 패키지 및 테스트동\n8층", specs:{floors:"8F",link:"M16 OHT"},
    processFlow:["다이싱","다이어태치","와이어본딩","몰딩","최종테스트"]},
  { name:"P&T5", x:780, z:660, w:130, d:90, h:48, color:0xc4956a, type:"패키지·테스트",
    detail:"P&T5 패키지 및 테스트동", specs:{floors:"4F"},
    processFlow:["다이싱","패키지조립","품질검증","출하대기"]},
];

const auxBuildings = [
  {name:"에너지센터",x:-350,z:340,w:50,d:38,h:38,color:0x889988},
  {name:"청운기숙사",x:-355,z:-110,w:55,d:42,h:52,color:0x8899aa},
  {name:"하나은행",x:400,z:360,w:45,d:35,h:32,color:0x998877},
  {name:"안내센터",x:-230,z:465,w:42,d:32,h:30,color:0x889988},
  {name:"복지관",x:290,z:465,w:45,d:34,h:35,color:0x998888},
  {name:"물류센터",x:-348,z:170,w:45,d:35,h:32,color:0x889988},
  {name:"연구동",x:-345,z:50,w:48,d:38,h:40,color:0x99887a},
  {name:"품질관리동",x:405,z:155,w:42,d:34,h:30,color:0x888899},
  {name:"체육관",x:-240,z:-120,w:45,d:35,h:38,color:0x887788},
  {name:"환경안전동",x:155,z:-120,w:42,d:34,h:30,color:0x888877},
];

export default function SKHynixCampus3D() {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const buildingMeshesRef = useRef([]);
  const startTimeRef = useRef(Date.now());
  const busGroupRef = useRef([]);
  const hwCarsRef = useRef([]);
  const chimneySmokesRef = useRef([]);

  const [selected, setSelected] = useState(null);
  const [tooltip, setTooltip] = useState(null);
  const [timeMode, setTimeMode] = useState("auto");
  const [bottleneck, setBottleneck] = useState(false);
  const [speechBubble, setSpeechBubble] = useState(null);
  const speechTimerRef = useRef(null);
  const bottleneckRef = useRef(false);
  const bottleneckWarningsRef = useRef([]);
  const timeModeRef = useRef("auto");

  const personDialogs = [
    '여기가 SK하이닉스 이천캠퍼스야~','오늘도 야근이다.. 힘내자!','회사 어때? 나쁘지 않아 ㅎㅎ',
    '밥 먹으러 가자~','M16동 진짜 크다..','여기 어디야? 처음 왔는데..',
    '클린룸 들어가기 전에 커피 한잔!','웨이퍼 수율 올려야 하는데..','OHT 물류 시스템 대단하지 않냐?',
    '이번 주 금요일이 제일 기다려진다','반도체가 미래다!','점심 뭐 먹을까? 된장찌개?',
    '오늘 날씨 좋다~','저기 정문 쪽에 카페 새로 생겼대','내일 교대근무라 일찍 자야해',
    '공정 교육 듣고 왔는데 어렵다..','이천에서 서울까지 버스 1시간이면 가','기숙사 밥 맛있어졌더라',
    '연봉 협상 잘 됐으면 좋겠다 ㅎ','DRAM 가격 올라가면 좋겠다~','여기 야경 진짜 이쁘다!',
    '주말에 뭐 하지? 이천 온천 갈까?','선배님 저 질문 있는데요..','신입사원인데 길을 잃었어요..',
    '에너지센터에서 커피 마시고 왔어','택배 왔다! 안내센터 가야지','운동하러 체육관 가자~',
    '오늘 공정 점검이래 바쁘겠다','M14동에서 M16동까지 OHT로 5분!','이천캠퍼스 최고 아니냐?!',
    '아 졸려.. 야근 끝나면 치킨 먹자','복지관에서 동호회 모임 있대','하이닉스 들어오길 잘했다 진짜',
    '여기 P&T4동이 8층이래 대박','안전모 쓰고 다녀야 해~','주차장 어디야? 맨날 헷갈려',
    '오늘 회의 몇 시지?','퇴근하고 치맥하자!','이 건물 뭐하는 데야?','고속도로 막히려나.. 일찍 가자',
  ];

  const sphericalRef = useRef({ theta: 1.57, phi: 0.85, radius: 420 });
  const targetRef = useRef(new THREE.Vector3(35, 0, 170));
  const dragRef = useRef({ isDragging: false, isRight: false, prev: { x: 0, y: 0 } });

  useEffect(() => { timeModeRef.current = timeMode; }, [timeMode]);

  const toggleBottleneck = () => {
    setBottleneck(prev => {
      const next = !prev;
      bottleneckRef.current = next;
      if (next) { window._startBottleneck?.(); }
      else { window._clearBottleneck?.(); }
      return next;
    });
  };

  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const W = container.clientWidth, H = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a12);
    scene.fog = new THREE.FogExp2(0x0a0a12, 0.0006);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, W / H, 1, 5000);
    camera.position.set(-200, 300, 450);
    camera.lookAt(250, 0, 250);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lights
    const ambient = new THREE.AmbientLight(0x667799, 0.9);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffeedd, 1.2);
    dirLight.position.set(300, 400, 200);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(2048, 2048);
    dirLight.shadow.camera.left = -500; dirLight.shadow.camera.right = 500;
    dirLight.shadow.camera.top = 500; dirLight.shadow.camera.bottom = -500;
    scene.add(dirLight);
    const hemi = new THREE.HemisphereLight(0x99bbdd, 0x445566, 0.8);
    scene.add(hemi);
    const pt = new THREE.PointLight(0x4488ff, 0.3, 600);
    pt.position.set(200, 150, 300);
    scene.add(pt);

    // Sun & Moon
    const sunMesh = new THREE.Mesh(new THREE.SphereGeometry(18,32,32), new THREE.MeshBasicMaterial({color:0xffdd44}));
    scene.add(sunMesh);
    const sunGlow = new THREE.Mesh(new THREE.SphereGeometry(25,32,32), new THREE.MeshBasicMaterial({color:0xffaa22,transparent:true,opacity:0.2}));
    sunMesh.add(sunGlow);
    const moonMesh = new THREE.Mesh(new THREE.SphereGeometry(20,32,32), new THREE.MeshBasicMaterial({color:0xfffff0}));
    scene.add(moonMesh);
    const moonGlow = new THREE.Mesh(new THREE.SphereGeometry(30,32,32), new THREE.MeshBasicMaterial({color:0xcceeff,transparent:true,opacity:0.25}));
    moonMesh.add(moonGlow);
    const moonLight = new THREE.PointLight(0x8899cc, 0, 800);
    moonMesh.add(moonLight);
    const skyCenter = new THREE.Vector3(35, 0, 170);
    const skyRadius = 350;

    // Ground
    const groundGeo = new THREE.PlaneGeometry(850, 800);
    const ground = new THREE.Mesh(groundGeo, new THREE.MeshStandardMaterial({color:0x141820,roughness:0.95}));
    ground.rotation.x = -Math.PI/2; ground.position.set(35,-0.5,170); ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(850, 42, 0x14171e, 0x10121a);
    grid.position.set(35, 0.05, 170);
    grid.material.transparent = true; grid.material.opacity = 0.07;
    scene.add(grid);

    // Roads helper
    function addRoad(x1,z1,x2,z2,width) {
      const dx=x2-x1,dz=z2-z1,len=Math.sqrt(dx*dx+dz*dz);
      const geo = new THREE.PlaneGeometry(len,width);
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color:0x252a35,roughness:0.75}));
      mesh.rotation.x=-Math.PI/2; mesh.position.set((x1+x2)/2,0.35,(z1+z2)/2);
      mesh.rotation.z=-Math.atan2(dz,dx); mesh.receiveShadow=true; scene.add(mesh);
      const cG = new THREE.PlaneGeometry(len,0.3);
      const cL = new THREE.Mesh(cG, new THREE.MeshStandardMaterial({color:0x556677,emissive:0x334455,emissiveIntensity:0.2}));
      cL.rotation.x=-Math.PI/2; cL.position.set((x1+x2)/2,0.38,(z1+z2)/2);
      cL.rotation.z=-Math.atan2(dz,dx); scene.add(cL);
    }

    // Main roads
    addRoad(R(60),RZ(270),R(1000),RZ(270),14);
    addRoad(R(60),RZ(470),R(1000),RZ(470),14);
    addRoad(R(60),RZ(680),R(1000),RZ(680),14);
    addRoad(R(130),RZ(100),R(130),RZ(880),12);
    addRoad(R(400),RZ(100),R(400),RZ(880),12);
    addRoad(R(680),RZ(100),R(680),RZ(880),12);
    addRoad(R(980),RZ(100),R(980),RZ(880),12);
    addRoad(R(60),RZ(100),R(1000),RZ(100),10);
    addRoad(R(60),RZ(880),R(1000),RZ(880),10);
    addRoad(R(60),RZ(100),R(60),RZ(880),10);
    addRoad(R(1000),RZ(100),R(1000),RZ(880),10);
    addRoad(-380,480,470,480,14);
    addRoad(-380,480,-380,-150,12);
    addRoad(470,480,470,-150,12);
    addRoad(-380,-150,470,-150,14);

    // Building creation
    const buildingMeshes = [];
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    function createBuilding(b) {
      const cx=(b.x+b.w/2-480)*SCALE, cz=(820-(b.z+b.d/2))*SCALE;
      const w=b.w*SCALE, d=b.d*SCALE, h=b.h*SCALE;
      const group = new THREE.Group();
      group.userData = {...b};
      const color = new THREE.Color(b.color);

      // Base
      const baseH = Math.min(h*0.15,5);
      const base = new THREE.Mesh(new THREE.BoxGeometry(w+2,baseH,d+2),
        new THREE.MeshStandardMaterial({color:0x2a2d35,roughness:0.9}));
      base.position.y=baseH/2; base.castShadow=true; base.receiveShadow=true; group.add(base);

      // Main body
      const mainH=h-baseH;
      const body = new THREE.Mesh(new THREE.BoxGeometry(w,mainH,d),
        new THREE.MeshStandardMaterial({color:color.clone().multiplyScalar(0.55),roughness:0.5,metalness:0.2}));
      body.position.y=baseH+mainH/2; body.castShadow=true; body.receiveShadow=true; group.add(body);

      // Panel lines
      const panelCnt = Math.floor(mainH/4);
      for(let i=0;i<panelCnt;i++){
        const py=baseH+2+i*4;
        const lm=new THREE.MeshStandardMaterial({color:color.clone().multiplyScalar(0.35),roughness:0.7});
        const lf=new THREE.Mesh(new THREE.BoxGeometry(w+0.2,0.15,0.15),lm);
        lf.position.set(0,py,d/2+0.1); group.add(lf);
        const lb=lf.clone(); lb.position.z=-d/2-0.1; group.add(lb);
      }

      // Windows
      const winSX=6,winW=3.5,winH=2.8;
      const wCF=Math.max(1,Math.floor((w-4)/winSX));
      const wCS=Math.max(1,Math.floor((d-4)/winSX));
      const wR=Math.max(1,Math.floor((mainH-3)/6));
      for(let row=0;row<wR;row++){
        const wy=baseH+3+row*6;
        for(let col=0;col<wCF;col++){
          const wx=-w/2+3+col*winSX;
          const wMat=new THREE.MeshStandardMaterial({
            color:0x8ac4ed,emissive:0x3388bb,emissiveIntensity:0.15+Math.random()*0.6,
            transparent:true,opacity:0.5+Math.random()*0.4,side:THREE.DoubleSide});
          const wF=new THREE.Mesh(new THREE.PlaneGeometry(winW,winH),wMat);
          wF.userData.isWindow=true; wF.position.set(wx,wy,d/2+0.15); group.add(wF);
          const wB=new THREE.Mesh(new THREE.PlaneGeometry(winW,winH),wMat.clone());
          wB.userData.isWindow=true; wB.position.set(wx,wy,-d/2-0.15); group.add(wB);
        }
        for(let col=0;col<wCS;col++){
          const wz=-d/2+3+col*winSX;
          const wMat=new THREE.MeshStandardMaterial({
            color:0x8ac4ed,emissive:0x3388bb,emissiveIntensity:0.15+Math.random()*0.6,
            transparent:true,opacity:0.5+Math.random()*0.4,side:THREE.DoubleSide});
          const wR2=new THREE.Mesh(new THREE.PlaneGeometry(winW,winH),wMat);
          wR2.userData.isWindow=true; wR2.rotation.y=Math.PI/2; wR2.position.set(w/2+0.15,wy,wz); group.add(wR2);
          const wL=new THREE.Mesh(new THREE.PlaneGeometry(winW,winH),wMat.clone());
          wL.userData.isWindow=true; wL.rotation.y=Math.PI/2; wL.position.set(-w/2-0.15,wy,wz); group.add(wL);
        }
      }

      // Roof
      const roof=new THREE.Mesh(new THREE.BoxGeometry(w+1.5,0.8,d+1.5),
        new THREE.MeshStandardMaterial({color:color.clone().multiplyScalar(0.4),roughness:0.6,metalness:0.3}));
      roof.position.y=h+0.4; roof.castShadow=true; group.add(roof);

      // Color strip
      const strip=new THREE.Mesh(new THREE.BoxGeometry(w+2,1.2,d+2),
        new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:0.5,roughness:0.3}));
      strip.position.y=h-0.5; group.add(strip);

      // Rooftop equipment
      const eqCnt=Math.floor(w/18);
      for(let i=0;i<eqCnt;i++){
        const eqW=3+Math.random()*3,eqH=2+Math.random()*3,eqD=3+Math.random()*2;
        const eq=new THREE.Mesh(new THREE.BoxGeometry(eqW,eqH,eqD),
          new THREE.MeshStandardMaterial({color:0x556677,roughness:0.6,metalness:0.4}));
        eq.position.set(-w/2+8+i*18,h+0.8+eqH/2,-d/4+Math.random()*d/2);
        eq.castShadow=true; group.add(eq);
      }

      // Door
      const doorW=Math.min(w*0.12,6);
      const door=new THREE.Mesh(new THREE.BoxGeometry(doorW,baseH*0.85,0.5),
        new THREE.MeshStandardMaterial({color:0x3a5577,emissive:0x223344,emissiveIntensity:0.3}));
      door.position.set(0,baseH*0.43,d/2+0.3); group.add(door);

      // Floor glow
      const glow=new THREE.Mesh(new THREE.PlaneGeometry(w+6,d+6),
        new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:0.12,transparent:true,opacity:0.25}));
      glow.rotation.x=-Math.PI/2; glow.position.y=0.2; group.add(glow);

      // Label (원본과 동일: 라운드 배경 + 하단 색상바 + 건물색 타입)
      const hexColor='#'+b.color.toString(16).padStart(6,'0');
      const lc=document.createElement("canvas");
      const ctx=lc.getContext("2d");
      const fs=32,tfs=16;
      ctx.font=`bold ${fs}px sans-serif`;
      const tw=ctx.measureText(b.name).width;
      ctx.font=`${tfs}px sans-serif`;
      const tw2=ctx.measureText(b.type).width;
      const cW=Math.max(tw,tw2)+40, cH=fs+tfs+28;
      lc.width=cW; lc.height=cH;
      // 라운드 배경
      ctx.fillStyle="rgba(8,8,16,0.75)";
      const rr=8;
      ctx.beginPath();
      ctx.moveTo(rr,0); ctx.lineTo(cW-rr,0); ctx.quadraticCurveTo(cW,0,cW,rr);
      ctx.lineTo(cW,cH-rr); ctx.quadraticCurveTo(cW,cH,cW-rr,cH);
      ctx.lineTo(rr,cH); ctx.quadraticCurveTo(0,cH,0,cH-rr);
      ctx.lineTo(0,rr); ctx.quadraticCurveTo(0,0,rr,0);
      ctx.closePath(); ctx.fill();
      // 하단 색상 바
      ctx.fillStyle=hexColor;
      ctx.fillRect(10,cH-5,cW-20,3);
      // 건물 이름
      ctx.font=`bold ${fs}px sans-serif`;
      ctx.fillStyle="#ffffff"; ctx.textAlign="center"; ctx.textBaseline="top";
      ctx.fillText(b.name,cW/2,8);
      // 타입 (건물 색상)
      ctx.font=`${tfs}px sans-serif`;
      ctx.fillStyle=hexColor;
      ctx.fillText(b.type,cW/2,fs+12);
      const tex=new THREE.CanvasTexture(lc); tex.minFilter=THREE.LinearFilter;
      const spriteScale=Math.max(w*0.55,22);
      const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));
      sp.scale.set(spriteScale,spriteScale*(cH/cW),1); sp.position.y=h+14; group.add(sp);
      // 라벨 연결선
      const lineGeo=new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0,h+1.5,0),new THREE.Vector3(0,h+10,0)]);
      const labelLine=new THREE.Line(lineGeo,new THREE.LineBasicMaterial({color:b.color,transparent:true,opacity:0.4}));
      group.add(labelLine);

      // Chimney
      if(b.chimneys){
        const chR=(b.chimneys.radius||8)*SCALE, chTR=chR*0.6, chH=(b.chimneys.height||50)*SCALE;
        const roofY=h+1.5, chX=w/2-chR-3;

        const pad=new THREE.Mesh(new THREE.BoxGeometry(chR*4,3,chR*4),
          new THREE.MeshStandardMaterial({color:0x3a3a44,roughness:0.9}));
        pad.position.set(chX,roofY+1.5,0); group.add(pad);

        const shaft=new THREE.Mesh(new THREE.CylinderGeometry(chTR,chR,chH,24),
          new THREE.MeshStandardMaterial({color:0x888899,roughness:0.4,metalness:0.35}));
        shaft.position.set(chX,roofY+3+chH/2,0); shaft.castShadow=true; group.add(shaft);

        [0.2,0.45,0.7,0.92].forEach((ratio,idx)=>{
          const sy=roofY+3+chH*ratio;
          const sr=chTR+(chR-chTR)*(1-ratio);
          const sG=new THREE.CylinderGeometry(sr+0.3,sr+0.3,2.5,24);
          const sM=new THREE.MeshStandardMaterial({
            color:idx%2===0?0xff2200:0xffffff,
            emissive:idx%2===0?0x550000:0x111111,emissiveIntensity:0.35});
          const s=new THREE.Mesh(sG,sM); s.position.set(chX,sy,0); group.add(s);
        });

        const topY=roofY+3+chH;
        const ring=new THREE.Mesh(new THREE.TorusGeometry(chTR+0.6,1,10,24),
          new THREE.MeshStandardMaterial({color:0xaaaaaa,roughness:0.3,metalness:0.6}));
        ring.rotation.x=Math.PI/2; ring.position.set(chX,topY,0); group.add(ring);

        const warn=new THREE.Mesh(new THREE.SphereGeometry(1.2,10,10),
          new THREE.MeshStandardMaterial({color:0xff0000,emissive:0xff0000,emissiveIntensity:1,transparent:true,opacity:0.9}));
        warn.position.set(chX,topY+2.5,0); warn.userData.isChimneyWarning=true; group.add(warn);
        const wPL=new THREE.PointLight(0xff2200,0.7,60);
        wPL.position.set(chX,topY+2.5,0); wPL.userData.isChimneyWarning=true; group.add(wPL);

        // Smoke
        const smokeN=30; const smokePos=new Float32Array(smokeN*3);
        for(let si=0;si<smokeN;si++){
          smokePos[si*3]=chX+(Math.random()-0.5)*chTR*2;
          smokePos[si*3+1]=topY+1+Math.random()*25;
          smokePos[si*3+2]=(Math.random()-0.5)*chTR*2;
        }
        const smokeGeo=new THREE.BufferGeometry();
        smokeGeo.setAttribute("position",new THREE.BufferAttribute(smokePos,3));
        const smokePts=new THREE.Points(smokeGeo,new THREE.PointsMaterial({
          color:0xcccccc,size:4,transparent:true,opacity:0.2,blending:THREE.AdditiveBlending,depthWrite:false}));
        smokePts.userData.smokeData={baseX:chX,baseY:topY+1,baseZ:0,count:smokeN,topRadius:chTR};
        group.add(smokePts);
        chimneySmokesRef.current.push(smokePts);
      }

      group.position.set(cx,-50,cz);
      group.userData.animDelay=Math.random()*2000;
      group.userData.animStarted=false;
      group.userData.animTime=0;
      scene.add(group);
      buildingMeshes.push(group);
    }

    buildings.forEach(b => createBuilding(b));

    // Aux buildings
    auxBuildings.forEach(ab=>{
      const g=new THREE.Group();
      g.userData={name:ab.name,type:"부속시설",detail:ab.name,color:ab.color};
      const geo=new THREE.BoxGeometry(ab.w,ab.h,ab.d);
      const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:ab.color,roughness:0.7,metalness:0.1}));
      mesh.position.y=ab.h/2; mesh.castShadow=true; g.add(mesh);
      const roofG=new THREE.BoxGeometry(ab.w+0.5,0.5,ab.d+0.5);
      const rf=new THREE.Mesh(roofG,new THREE.MeshStandardMaterial({color:0x444c55,roughness:0.6}));
      rf.position.y=ab.h+0.25; g.add(rf);
      // label
      const lc2=document.createElement("canvas"); const cx2=lc2.getContext("2d");
      cx2.font="600 22px sans-serif"; const tw2=cx2.measureText(ab.name).width;
      lc2.width=tw2+24;lc2.height=34;
      cx2.fillStyle="rgba(10,10,20,0.7)"; cx2.fillRect(0,0,lc2.width,lc2.height);
      cx2.font="600 22px sans-serif"; cx2.fillStyle="#ccddee";
      cx2.textAlign="center"; cx2.textBaseline="middle";
      cx2.fillText(ab.name,lc2.width/2,lc2.height/2);
      const t2=new THREE.CanvasTexture(lc2); t2.minFilter=THREE.LinearFilter;
      const sp2=new THREE.Sprite(new THREE.SpriteMaterial({map:t2,transparent:true,depthTest:false}));
      sp2.scale.set(14,14*(lc2.height/lc2.width),1); sp2.position.y=ab.h+8; g.add(sp2);
      g.position.set(ab.x,0,ab.z);
      scene.add(g); buildingMeshes.push(g);
    });

    // Outer buildings
    const outerColors=[0x667788,0x778899,0x606872,0x7a8090,0x8b8f98,0x6e7580];
    const outerData=[
      [-290,300,55,40,60],[-300,220,48,38,50],[-285,140,60,45,75],[-295,55,50,40,45],[-288,-20,55,42,58],[-300,-85,45,35,40],
      [340,300,50,40,55],[350,215,55,42,70],[335,130,48,40,52],[348,45,45,38,42],[340,-30,52,40,65],[345,-95,48,36,48],
      [-155,-100,52,40,50],[-65,-108,48,36,58],[35,-98,45,40,44],[120,-105,55,38,52],[210,-100,48,36,40]];
    outerData.forEach(([ox,oz,ow,od,oh],i)=>{
      const c=outerColors[i%outerColors.length];
      const m=new THREE.Mesh(new THREE.BoxGeometry(ow,oh,od),new THREE.MeshStandardMaterial({color:c,roughness:0.85}));
      m.position.set(ox,oh/2,oz); m.castShadow=true; m.receiveShadow=true; scene.add(m);
    });

    buildingMeshesRef.current = buildingMeshes;

    // OHT Bridges
    function createOHTBridge(sx,sz,ex,ez,bHeight,label) {
      const dx=ex-sx,dz=ez-sz,len=Math.sqrt(dx*dx+dz*dz);
      if(len<1) return;
      const angle=Math.atan2(dz,dx), bH=2.5, bW=2.8;
      const g=new THREE.Group();
      // 바닥판
      const floorM=new THREE.MeshStandardMaterial({color:0x556677,roughness:0.6,metalness:0.3});
      const fl=new THREE.Mesh(new THREE.BoxGeometry(len,0.3,bW),floorM);
      fl.position.y=-bH/2; g.add(fl);
      // 천장판
      const cl=new THREE.Mesh(new THREE.BoxGeometry(len,0.25,bW+0.4),floorM.clone());
      cl.position.y=bH/2; g.add(cl);
      // 프레임 기둥 + 유리
      const seg=Math.max(1,Math.floor(len/6));
      const fMat=new THREE.MeshStandardMaterial({color:0x88999a,roughness:0.4,metalness:0.5});
      for(let i=0;i<=seg;i++){
        const px=-len/2+i*(len/seg);
        const fL=new THREE.Mesh(new THREE.BoxGeometry(0.25,bH,0.25),fMat);
        fL.position.set(px,0,bW/2); g.add(fL);
        const fR=fL.clone(); fR.position.z=-bW/2; g.add(fR);
      }
      const gMat=new THREE.MeshStandardMaterial({color:0x99ccee,emissive:0x224466,emissiveIntensity:0.15,transparent:true,opacity:0.25,side:THREE.DoubleSide});
      for(let i=0;i<seg;i++){
        const px=-len/2+(i+0.5)*(len/seg), pw=(len/seg)-0.5;
        const gL=new THREE.Mesh(new THREE.PlaneGeometry(pw,bH-0.6),gMat);
        gL.position.set(px,0,bW/2+0.05); g.add(gL);
        const gR=new THREE.Mesh(new THREE.PlaneGeometry(pw,bH-0.6),gMat.clone());
        gR.position.set(px,0,-bW/2-0.05); g.add(gR);
      }
      // OHT 레일
      const rG=new THREE.BoxGeometry(len-1,0.2,0.3);
      const rMat=new THREE.MeshStandardMaterial({color:0xccaa22,emissive:0xaa8800,emissiveIntensity:0.3,roughness:0.3,metalness:0.5});
      const r1=new THREE.Mesh(rG,rMat); r1.position.set(0,bH/2-0.3,-0.5); g.add(r1);
      const r2=new THREE.Mesh(rG,rMat.clone()); r2.position.set(0,bH/2-0.3,0.5); g.add(r2);
      // 하부 발광
      const gl=new THREE.Mesh(new THREE.BoxGeometry(len,0.15,0.6),
        new THREE.MeshStandardMaterial({color:0x44aaee,emissive:0x2288cc,emissiveIntensity:0.5,transparent:true,opacity:0.5}));
      gl.position.y=-bH/2-0.2; g.add(gl);
      // 지지 기둥
      const pCnt=Math.max(1,Math.floor(len/35));
      for(let i=0;i<=pCnt;i++){
        const px=-len/2+5+i*((len-10)/Math.max(1,pCnt));
        const p=new THREE.Mesh(new THREE.BoxGeometry(1.2,bHeight,1.2),
          new THREE.MeshStandardMaterial({color:0x667788,roughness:0.6,metalness:0.3}));
        p.castShadow=true; p.position.set(px,-bH/2-bHeight/2,0); g.add(p);
      }
      if(label){
        const lc=document.createElement("canvas"); const lx=lc.getContext("2d");
        lx.font="bold 20px sans-serif"; const tw=lx.measureText(label).width;
        lc.width=tw+20;lc.height=30;
        lx.fillStyle="rgba(0,30,60,0.75)"; lx.fillRect(0,0,lc.width,lc.height);
        lx.fillStyle="rgba(0,150,220,0.8)"; lx.fillRect(0,lc.height-2,lc.width,2);
        lx.font="bold 20px sans-serif"; lx.fillStyle="#ddeeff";
        lx.textAlign="center"; lx.textBaseline="middle";
        lx.fillText(label,lc.width/2,lc.height/2);
        const tex=new THREE.CanvasTexture(lc); tex.minFilter=THREE.LinearFilter;
        const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));
        sp.scale.set(12,12*(lc.height/lc.width),1); sp.position.y=bH/2+4; g.add(sp);
      }
      g.position.set((sx+ex)/2,bHeight,(sz+ez)/2);
      g.rotation.y=-angle; scene.add(g);
    }

    function getBldCenter(b){return{x:(b.x+b.w/2-480)*SCALE,z:(820-(b.z+b.d/2))*SCALE,h:b.h*SCALE}}
    const m14=getBldCenter(buildings[0]),m10a=getBldCenter(buildings[1]),m16=getBldCenter(buildings[4]);
    const pt4=getBldCenter(buildings[7]),pt5=getBldCenter(buildings[8]);
    const ohtH1=Math.min(m14.h,m16.h)*0.7;

    // M14↔M10A
    // M14↔M10A (ㄷ자 경로) - 건물 base(w+2)보다 바깥으로 시작
    // M14 half-width=52.5, base half=53.5 → 최소 54 이상 오프셋
    // M10A half-width=50, base half=51 → 최소 52 이상 오프셋
    const m14_3F=m14.h*3/8,m10a_2F=m10a.h*2/5,ohtHm=(m14_3F+m10a_2F)/2;
    const m14L=m14.x-56,outL=m14L-28,m10aL=m10a.x-54;
    createOHTBridge(m14L,m14.z,outL,m14.z,ohtHm,"리프터(3층)");
    createOHTBridge(outL,m14.z,outL,m10a.z,ohtHm,"OHT 레일");
    createOHTBridge(outL,m10a.z,m10aL,m10a.z,ohtHm,"리프터(2층)");

    // M14↔M16 (M10B 왼쪽 우회)
    const safeX=-15,m16LX=25;
    createOHTBridge(m14.x+55,m14.z,safeX,m14.z,ohtH1,"컨베이어");
    createOHTBridge(safeX,m14.z,safeX,m16.z+10,ohtH1,"OHT 레일");
    createOHTBridge(safeX,m16.z+10,m16LX,m16.z+10,ohtH1,"리프터");

    // M16↔P&T4
    const m16_3F=m16.h*3/11,pt4_5F=pt4.h*5/8,ohtH2=(m16_3F+pt4_5F)/2;
    const m16R=m16.x+55,cX2=pt4.x-30;
    createOHTBridge(m16R,m16.z,cX2,m16.z,ohtH2,"컨베이어(3층)");
    createOHTBridge(cX2,m16.z,cX2,pt4.z+10,ohtH2,"리프터(5층)");

    // M16↔P&T5
    const m16_2F=m16.h*2/11,pt5_3F=pt5.h*3/5,ohtH3=(m16_2F+pt5_3F)/2;
    const fRX=pt4.x+10,pt5TZ=pt5.z-10;
    createOHTBridge(m16R,m16.z-15,fRX,m16.z-15,ohtH3,"컨베이어(2층)");
    createOHTBridge(fRX,m16.z-15,fRX,pt5TZ,ohtH3,"리프터");
    createOHTBridge(fRX,pt5TZ,pt5.x+30,pt5TZ,ohtH3,"컨베이어(3층)");

    // Trees
    const trunkMat=new THREE.MeshStandardMaterial({color:0x5a4030,roughness:0.9});
    const trunkGeo=new THREE.CylinderGeometry(0.12,0.22,1,5);
    const coneGeo=new THREE.ConeGeometry(1,1,6);
    const sphereGeo=new THREE.SphereGeometry(1,6,5);
    const leafCs=[0x1e5a18,0x2d6a22,0x1a5010,0x2a6830,0x1d4a12];

    function addTree(x,z){
      const s=2+Math.random()*2.5, g=new THREE.Group(), ic=Math.random()>0.4;
      const lc=leafCs[Math.floor(Math.random()*5)];
      const lm=new THREE.MeshStandardMaterial({color:lc,roughness:0.8,flatShading:true});
      const trunk=new THREE.Mesh(trunkGeo,trunkMat);
      trunk.scale.set(s*0.5,s*2.5,s*0.5); trunk.position.y=s*1.2; g.add(trunk);
      if(ic){
        for(let i=0;i<2+(Math.random()>0.5?1:0);i++){
          const cs=s*(1.4-i*0.3);
          const cone=new THREE.Mesh(coneGeo,lm);
          cone.scale.set(cs,cs*1.2,cs); cone.position.y=s*2.2+i*s*0.9; g.add(cone);
        }
      }else{
        const crown=new THREE.Mesh(sphereGeo,lm);
        crown.scale.set(s*1.3,s*1.1,s*1.3); crown.position.y=s*3.2; g.add(crown);
      }
      g.position.set(x,0,z); scene.add(g);
    }

    for(let i=0;i<18;i++) addTree(-230+Math.random()*20,340-i*35+Math.random()*12);
    for(let i=0;i<18;i++) addTree(275+Math.random()*20,340-i*35+Math.random()*12);
    for(let i=0;i<8;i++) addTree(-200+i*20+Math.random()*8,380+Math.random()*15);
    for(let i=0;i<8;i++) addTree(100+i*20+Math.random()*8,380+Math.random()*15);
    for(let i=0;i<14;i++) addTree(-180+i*32+Math.random()*10,-45-Math.random()*20);

    // People
    const bodyColors=[0x2244aa,0x3366cc,0xeeeeee,0x333333,0xcc4444,0x44aa44,0x225588,0x884422];
    const skinColors=[0xffccaa,0xf5c09a,0xeebb88,0xdda878];
    const pHeadGeo=new THREE.SphereGeometry(1,5,4);
    const pBodyGeo=new THREE.BoxGeometry(1,1,1);
    const zones=[{xMin:-190,xMax:240,zMin:-30,zMax:350,count:50},{xMin:-30,xMax:60,zMin:350,zMax:400,count:10}];
    zones.forEach(zone=>{
      for(let i=0;i<zone.count;i++){
        const px=zone.xMin+Math.random()*(zone.xMax-zone.xMin);
        const pz=zone.zMin+Math.random()*(zone.zMax-zone.zMin);
        const sc=0.7+Math.random()*0.3;
        const g=new THREE.Group();
        const head=new THREE.Mesh(pHeadGeo,new THREE.MeshStandardMaterial({color:skinColors[Math.floor(Math.random()*4)]}));
        head.scale.set(sc*0.8,sc*0.9,sc*0.8); head.position.y=sc*5.2; g.add(head);
        const bd=new THREE.Mesh(pBodyGeo,new THREE.MeshStandardMaterial({color:bodyColors[Math.floor(Math.random()*8)]}));
        bd.scale.set(sc*1.6,sc*2.8,sc*1); bd.position.y=sc*3.2; g.add(bd);
        const legM=new THREE.MeshStandardMaterial({color:0x333344}); 
        const lL=new THREE.Mesh(pBodyGeo,legM); lL.scale.set(sc*0.55,sc*2.2,sc*0.55); lL.position.set(-sc*0.35,sc*0.9,0); g.add(lL);
        const lR=new THREE.Mesh(pBodyGeo,legM.clone()); lR.scale.set(sc*0.55,sc*2.2,sc*0.55); lR.position.set(sc*0.35,sc*0.9,0); g.add(lR);
        g.position.set(px,0,pz); g.userData.isPerson=true;
        g.userData.walkSpeed=0.02+Math.random()*0.04;
        g.userData.walkDir=Math.random()*Math.PI*2;
        g.userData.walkTimer=Math.random()*100;
        g.userData.zone=zone;
        scene.add(g);
      }
    });

    // Highway
    const hwX=520,laneW=9,medW=2.5;
    for(let side of[-1,1]){
      const rx=hwX+side*(laneW/2+medW/2);
      const rd=new THREE.Mesh(new THREE.PlaneGeometry(laneW,700),new THREE.MeshStandardMaterial({color:0x2c3040,roughness:0.7}));
      rd.rotation.x=-Math.PI/2; rd.position.set(rx,0.4,170); rd.receiveShadow=true; scene.add(rd);
    }
    const med=new THREE.Mesh(new THREE.BoxGeometry(medW,1.8,700),new THREE.MeshStandardMaterial({color:0x555d68,roughness:0.8}));
    med.position.set(hwX,0.9,170); scene.add(med);
    const wall=new THREE.Mesh(new THREE.BoxGeometry(1.2,8,700),new THREE.MeshStandardMaterial({color:0x667766,roughness:0.7}));
    wall.position.set(hwX-laneW-medW/2-5,4,170); wall.castShadow=true; scene.add(wall);
    // HW Label
    const hwLC=document.createElement("canvas"); const hwCx=hwLC.getContext("2d");
    hwCx.font="bold 28px sans-serif"; const hwTW=hwCx.measureText("중부고속도로").width;
    hwLC.width=hwTW+30;hwLC.height=42;
    hwCx.fillStyle="rgba(0,60,20,0.8)";hwCx.fillRect(0,0,hwLC.width,hwLC.height);
    hwCx.font="bold 28px sans-serif";hwCx.fillStyle="#ffffff";hwCx.textAlign="center";hwCx.textBaseline="middle";
    hwCx.fillText("중부고속도로",hwLC.width/2,hwLC.height/2);
    const hwTex=new THREE.CanvasTexture(hwLC); hwTex.minFilter=THREE.LinearFilter;
    const hwSp=new THREE.Sprite(new THREE.SpriteMaterial({map:hwTex,transparent:true,depthTest:false}));
    hwSp.scale.set(22,22*(hwLC.height/hwLC.width),1); hwSp.position.set(hwX,18,170); scene.add(hwSp);

    // Highway cars
    const carColors=[0xcc2222,0x2244cc,0xeeeeee,0x333333,0x44aa44,0xddaa22,0x8844aa,0x22aacc];
    const hwCars=[];
    for(let i=0;i<10;i++){
      const g=new THREE.Group();
      const isUp=i<5;
      const lO=isUp?-(laneW/2+medW/2):(laneW/2+medW/2);
      const cc=carColors[i%8];
      const bG=new THREE.Mesh(new THREE.BoxGeometry(3,2.2,6),new THREE.MeshStandardMaterial({color:cc,roughness:0.3,metalness:0.3}));
      bG.position.y=1.8; g.add(bG);
      const cab=new THREE.Mesh(new THREE.BoxGeometry(2.4,1.6,3),new THREE.MeshStandardMaterial({color:cc,roughness:0.3}));
      cab.position.set(0,3.2,-0.3); g.add(cab);
      g.position.set(hwX+lO+(Math.random()-0.5)*3,0,-180+Math.random()*700);
      g.userData.speed=0.3+Math.random()*0.5; g.userData.dir=isUp?1:-1;
      scene.add(g); hwCars.push(g);
    }
    hwCarsRef.current=hwCars;

    // Shuttle buses
    const buses=[];
    function createBus(color,routeP){
      const g=new THREE.Group();
      const bd=new THREE.Mesh(new THREE.BoxGeometry(5,4.5,14),new THREE.MeshStandardMaterial({color,roughness:0.4,metalness:0.2}));
      bd.position.y=3.5; bd.castShadow=true; g.add(bd);
      const rf=new THREE.Mesh(new THREE.BoxGeometry(5.2,0.5,14.2),new THREE.MeshStandardMaterial({color:0xeeeeee}));
      rf.position.y=5.8; g.add(rf);
      g.userData.route=routeP; g.userData.progress=0; g.userData.speed=0.4; g.userData.forward=1;
      g.position.set(routeP[0].x,0,routeP[0].z);
      scene.add(g); buses.push(g);
    }
    createBus(0x2266aa,[{x:-190,z:RZ(470)},{x:240,z:RZ(470)}]);
    createBus(0x44aa44,[{x:R(400),z:340},{x:R(400),z:-30}]);
    busGroupRef.current=buses;

    // Gate
    for(let side of[-1,1]){
      const pillar=new THREE.Mesh(new THREE.BoxGeometry(3,15,3),new THREE.MeshStandardMaterial({color:0x992211,emissive:0x551100,emissiveIntensity:0.3}));
      pillar.position.set(15+side*15,7.5,420); pillar.castShadow=true; scene.add(pillar);
    }
    const arch=new THREE.Mesh(new THREE.BoxGeometry(35,2,3),new THREE.MeshStandardMaterial({color:0xcc3322,emissive:0x661100,emissiveIntensity:0.3}));
    arch.position.set(15,15,420); scene.add(arch);
    const gLC=document.createElement("canvas");const gCx=gLC.getContext("2d");
    gCx.font="bold 36px sans-serif";const gTW=gCx.measureText("SK hynix").width;
    gLC.width=gTW+30;gLC.height=48;
    gCx.font="bold 36px sans-serif";gCx.fillStyle="#e74c3c";gCx.textAlign="center";gCx.textBaseline="middle";
    gCx.fillText("SK",gLC.width/2-gCx.measureText("hynix").width/2-5,gLC.height/2);
    gCx.fillStyle="#ffffff";gCx.fillText("hynix",gLC.width/2+20,gLC.height/2);
    const gTex=new THREE.CanvasTexture(gLC); gTex.minFilter=THREE.LinearFilter;
    const gSp=new THREE.Sprite(new THREE.SpriteMaterial({map:gTex,transparent:true,depthTest:false}));
    gSp.scale.set(18,18*(gLC.height/gLC.width),1); gSp.position.set(15,20,421); scene.add(gSp);

    // Camera update
    function updateCamera(){
      const s=sphericalRef.current,t=targetRef.current;
      camera.position.set(
        t.x+s.radius*Math.sin(s.phi)*Math.cos(s.theta),
        t.y+s.radius*Math.cos(s.phi),
        t.z+s.radius*Math.sin(s.phi)*Math.sin(s.theta));
      camera.lookAt(t);
    }

    // Sky update
    function updateSky(elapsed){
      let hours;
      const tm=timeModeRef.current;
      if(tm==="morning") hours=10;
      else if(tm==="night") hours=0;
      else{const now=new Date();hours=now.getHours()+now.getMinutes()/60;}

      const sunAngle=((hours-6)/12)*Math.PI;
      const sunUp=hours>=5.5&&hours<=18.5;
      if(sunUp){
        const sx=skyCenter.x+skyRadius*Math.cos(sunAngle);
        const sy=skyRadius*Math.sin(sunAngle)*0.7;
        sunMesh.position.set(sx,Math.max(sy,-20),skyCenter.z-200);
        sunMesh.visible=true;
      }else sunMesh.visible=false;

      const moonAngle=((hours-18+24)%24/12)*Math.PI;
      const moonUp=hours>=17.5||hours<=6.5;
      if(moonUp){
        const mx=skyCenter.x-skyRadius*Math.cos(moonAngle);
        const my=skyRadius*Math.sin(moonAngle)*0.6;
        moonMesh.position.set(mx,Math.max(my,-20),skyCenter.z+150);
        moonMesh.visible=true;
        const alt=Math.max(0,my)/(skyRadius*0.6);
        moonLight.intensity=2.5*alt;
      }else{moonMesh.visible=false;moonLight.intensity=0;}

      let skyP;
      if(hours>=6&&hours<=18) skyP=1-Math.abs(hours-12)/6;
      else skyP=0;
      if(hours>=5&&hours<6) skyP=(hours-5)*0.3;
      if(hours>18&&hours<=19) skyP=(19-hours)*0.3;

      const nR=0x10/255,nG=0x12/255,nB=0x22/255;
      const dR=0x55/255,dG=0x99/255,dB=0xdd/255;
      let r,g2,b2;
      if(skyP>0.3){const t2=(skyP-0.3)/0.7;r=dR*t2+nR*(1-t2);g2=dG*t2+nG*(1-t2);b2=dB*t2+nB*(1-t2);}
      else{r=nR;g2=nG;b2=nB;}
      scene.background.setRGB(r,g2,b2);
      scene.fog.color.setRGB(r,g2,b2);
      ambient.intensity=0.8+(1.5-0.8)*skyP;
      dirLight.intensity=0.5+(1.8-0.5)*skyP;
      if(sunUp){dirLight.position.copy(sunMesh.position);dirLight.color.setHex(skyP>0.3?0xffeedd:0xff8844);}
      else{dirLight.position.set(100,300,100);dirLight.color.setHex(0x334466);}
      renderer.toneMappingExposure=1.2+skyP*0.6;

      const isNight=skyP<0.15;
      buildingMeshes.forEach(grp=>{
        grp.children.forEach(child=>{
          if(child.userData?.isWindow){
            if(isNight){
              child.material.color.setHex(0xffdd88);child.material.emissive.setHex(0xddaa33);
              child.material.emissiveIntensity=0.6+Math.random()*0.35;
            }else{
              child.material.color.setHex(0x8ac4ed);child.material.emissive.setHex(0x3388bb);
              child.material.emissiveIntensity=0.03;
            }
          }
        });
      });
    }

    // Animate
    let animId;
    function animate(){
      animId=requestAnimationFrame(animate);
      const elapsed=Date.now()-startTimeRef.current;

      // Building rise
      buildingMeshes.forEach(grp=>{
        if(elapsed>grp.userData.animDelay){
          if(!grp.userData.animStarted){grp.userData.animStarted=true;grp.userData.animTime=0;}
          grp.userData.animTime+=0.03;
          const t=Math.min(1,grp.userData.animTime);
          grp.position.y=-50+50*(1-Math.pow(1-t,3));
        }
      });

      updateSky(elapsed);

      // Window flicker
      if(Math.random()<0.02){
        const rB=buildingMeshes[Math.floor(Math.random()*buildingMeshes.length)];
        rB.children.forEach(c=>{if(c.userData?.isWindow)c.material.emissiveIntensity=0.3+Math.random()*0.6;});
      }

      // People walking
      scene.children.forEach(obj=>{
        if(obj.userData?.isPerson){
          obj.userData.walkTimer+=1;
          if(obj.userData.walkTimer%200<1) obj.userData.walkDir+=(Math.random()-0.5)*1.5;
          const spd=obj.userData.walkSpeed,dir=obj.userData.walkDir;
          obj.position.x+=Math.cos(dir)*spd; obj.position.z+=Math.sin(dir)*spd;
          obj.rotation.y=-dir+Math.PI/2;
          const z2=obj.userData.zone;
          if(obj.position.x<z2.xMin){obj.position.x=z2.xMin;obj.userData.walkDir=Math.random()*Math.PI;}
          if(obj.position.x>z2.xMax){obj.position.x=z2.xMax;obj.userData.walkDir=Math.PI+Math.random()*Math.PI;}
          if(obj.position.z<z2.zMin){obj.position.z=z2.zMin;obj.userData.walkDir=Math.PI*0.5*Math.random();}
          if(obj.position.z>z2.zMax){obj.position.z=z2.zMax;obj.userData.walkDir=-Math.PI*0.5+Math.random()*Math.PI;}
          const swing=Math.sin(obj.userData.walkTimer*0.15)*0.15;
          if(obj.children[2])obj.children[2].rotation.x=swing;
          if(obj.children[3])obj.children[3].rotation.x=-swing;
        }
      });

      // Buses
      busGroupRef.current.forEach(bus=>{
        const rr=bus.userData.route,p0=rr[0],p1=rr[1];
        bus.userData.progress+=bus.userData.speed*0.002*bus.userData.forward;
        if(bus.userData.progress>=1){bus.userData.progress=1;bus.userData.forward=-1;}
        if(bus.userData.progress<=0){bus.userData.progress=0;bus.userData.forward=1;}
        const t=bus.userData.progress;
        bus.position.x=p0.x+(p1.x-p0.x)*t;
        bus.position.z=p0.z+(p1.z-p0.z)*t;
        const ddx=(p1.x-p0.x)*bus.userData.forward,ddz=(p1.z-p0.z)*bus.userData.forward;
        bus.rotation.y=-Math.atan2(ddz,ddx)+Math.PI/2;
      });

      // Highway cars
      hwCarsRef.current.forEach(car=>{
        car.position.z+=car.userData.speed*car.userData.dir;
        if(car.position.z>530)car.position.z=-190;
        if(car.position.z<-190)car.position.z=530;
      });

      // Chimney smoke
      chimneySmokesRef.current.forEach(smoke=>{
        const pos=smoke.geometry.attributes.position.array;
        const sd=smoke.userData.smokeData,tr=sd.topRadius||2;
        for(let si=0;si<sd.count;si++){
          pos[si*3]+=(Math.random()-0.5)*0.1+Math.sin(elapsed*0.0008+si)*0.04;
          pos[si*3+1]+=0.1+Math.random()*0.06;
          pos[si*3+2]+=(Math.random()-0.5)*0.08;
          if(pos[si*3+1]>sd.baseY+25){
            pos[si*3]=sd.baseX+(Math.random()-0.5)*tr*2;
            pos[si*3+1]=sd.baseY;
            pos[si*3+2]=sd.baseZ+(Math.random()-0.5)*tr*2;
          }
        }
        smoke.geometry.attributes.position.needsUpdate=true;
        smoke.material.opacity=0.15+Math.sin(elapsed*0.003)*0.05;
      });

      // Chimney warning blink
      buildingMeshes.forEach(grp=>{
        grp.traverse(child=>{
          if(child.userData?.isChimneyWarning){
            const blink=Math.sin(elapsed*0.006)>0?1:0.1;
            if(child.isMesh){child.material.emissiveIntensity=blink;child.material.opacity=0.3+blink*0.7;}
            else if(child.isLight)child.intensity=blink*0.4;
          }
        });
      });

      // Bottleneck flicker
      if(bottleneckRef.current){
        const bt=Date.now()*0.005;
        buildingMeshes.forEach(grp=>{
          grp.traverse(child=>{
            if(child.userData?._isBottleneck&&child.isMesh&&child.material.emissive){
              const flicker=0.3+Math.sin(bt+Math.random())*0.3;
              child.material.emissive.setHex(0xff2200);
              child.material.emissiveIntensity=flicker;
              child.material.color.lerp(new THREE.Color(0xff3311),0.15);
            }
          });
        });
        bottleneckWarningsRef.current.forEach(w=>{
          if(w.light) w.light.intensity=1.0+Math.sin(bt*2)*1.5;
          if(w.sprite){
            w.sprite.material.opacity=0.7+Math.sin(bt*3)*0.3;
          }
        });
        // Gossip bubbles position update
        const bDoms=document.querySelectorAll('.bneck-bubble');
        let bIdx=0;
        scene.children.forEach(obj=>{
          if(obj.userData?.isPerson&&obj.userData?.hasGossip&&bIdx<bDoms.length){
            const pos3=new THREE.Vector3();
            obj.getWorldPosition(pos3); pos3.y+=8;
            pos3.project(camera);
            const sx2=(pos3.x*0.5+0.5)*container.clientWidth;
            const sy2=(-pos3.y*0.5+0.5)*container.clientHeight;
            bDoms[bIdx].style.left=sx2+'px';
            bDoms[bIdx].style.top=(sy2-30)+'px';
            bIdx++;
          }
        });
      }

      renderer.render(scene,camera);
    }

    // Bottleneck start/clear functions
    function startBottleneckFn(){
      const targetNames=['M14A/B','M16A/B'];
      const targets=buildingMeshes.filter(g=>targetNames.includes(g.userData.name));
      targets.forEach(group=>{
        group.traverse(child=>{
          if(child.isMesh&&child.material){
            child.userData._origColor=child.material.color.clone();
            child.userData._origEmissive=child.material.emissive?child.material.emissive.clone():null;
            child.userData._origEmissiveIntensity=child.material.emissiveIntensity||0;
            child.userData._isBottleneck=true;
          }
        });
        // Warning sprite
        const bH=group.userData.h*SCALE;
        const wC=document.createElement('canvas');
        wC.width=512;wC.height=256;
        const wCtx=wC.getContext('2d');
        wCtx.fillStyle='rgba(180,20,10,0.85)';
        wCtx.beginPath();wCtx.roundRect(10,10,492,236,16);wCtx.fill();
        wCtx.strokeStyle='#ff3300';wCtx.lineWidth=4;
        wCtx.beginPath();wCtx.roundRect(10,10,492,236,16);wCtx.stroke();
        wCtx.fillStyle='#ffcc00';wCtx.font='bold 60px sans-serif';wCtx.textAlign='center';
        wCtx.fillText('⚠',256,80);
        wCtx.fillStyle='#ffffff';wCtx.font='bold 36px sans-serif';
        wCtx.fillText('QUE 병목 예측',256,140);
        wCtx.fillStyle='#ff8866';wCtx.font='24px sans-serif';
        wCtx.fillText(group.userData.name+' 대기열 과부하',256,185);
        wCtx.fillStyle='#ffaa88';wCtx.font='18px sans-serif';
        wCtx.fillText('처리량 초과 · 지연 발생 예상',256,220);
        const wTex=new THREE.CanvasTexture(wC);
        const wSp=new THREE.Sprite(new THREE.SpriteMaterial({map:wTex,transparent:true,depthTest:false}));
        wSp.scale.set(60,30,1);wSp.position.set(0,bH+40,0);
        group.add(wSp);
        bottleneckWarningsRef.current.push({sprite:wSp,group});
        const fL=new THREE.PointLight(0xff2200,2.0,100);
        fL.position.set(0,bH+5,0);group.add(fL);
        bottleneckWarningsRef.current.push({light:fL,group});
      });
      // Gossip bubbles on people
      const gossips=['M14 QUE 병목이 예측된대...','M16도 대기열 과부하래!','병목 심하면 물류 멈춘다던데...','OHT 지연되면 큰일이야;;','QUE 터지면 라인 스톱이래...'];
      let pIdx=0;
      scene.children.forEach(obj=>{
        if(obj.userData?.isPerson&&pIdx<5){
          obj.userData.hasGossip=true;
          const bubble=document.createElement('div');
          bubble.className='bneck-bubble';
          bubble.style.cssText='position:fixed;z-index:350;pointer-events:none;background:rgba(180,20,10,0.9);border:1px solid #ff4444;border-radius:12px;padding:8px 14px;font-size:11px;color:#ffcccc;font-weight:600;white-space:nowrap;box-shadow:0 0 15px rgba(255,50,30,0.4);display:none;font-family:inherit;';
          bubble.textContent=gossips[pIdx%5];
          document.body.appendChild(bubble);
          setTimeout(()=>{bubble.style.display='block';},pIdx*800);
          pIdx++;
        }
      });
    }

    function clearBottleneckFn(){
      bottleneckWarningsRef.current.forEach(w=>{
        if(w.sprite)w.group.remove(w.sprite);
        if(w.light)w.group.remove(w.light);
      });
      bottleneckWarningsRef.current=[];
      buildingMeshes.forEach(grp=>{
        grp.traverse(child=>{
          if(child.userData?._isBottleneck&&child.isMesh){
            if(child.userData._origColor)child.material.color.copy(child.userData._origColor);
            if(child.userData._origEmissive)child.material.emissive.copy(child.userData._origEmissive);
            child.material.emissiveIntensity=child.userData._origEmissiveIntensity||0;
            delete child.userData._isBottleneck;
            delete child.userData._origColor;
            delete child.userData._origEmissive;
            delete child.userData._origEmissiveIntensity;
          }
        });
      });
      document.querySelectorAll('.bneck-bubble').forEach(el=>el.remove());
      scene.children.forEach(obj=>{if(obj.userData?.hasGossip)delete obj.userData.hasGossip;});
    }

    // Expose bottleneck functions
    window._startBottleneck=startBottleneckFn;
    window._clearBottleneck=clearBottleneckFn;

    setTimeout(()=>{startTimeRef.current=Date.now();animate();},100);

    // Events
    const el=renderer.domElement;
    const onDown=e=>{
      if(e.button===2)dragRef.current.isRight=true;
      else dragRef.current.isDragging=true;
      dragRef.current.prev={x:e.clientX,y:e.clientY};
    };
    const onMove=e=>{
      const d=dragRef.current,dx=e.clientX-d.prev.x,dy=e.clientY-d.prev.y;
      if(d.isDragging){
        sphericalRef.current.theta-=dx*0.005;
        sphericalRef.current.phi=Math.max(0.2,Math.min(1.5,sphericalRef.current.phi-dy*0.005));
        updateCamera();
      }
      if(d.isRight){
        const right=new THREE.Vector3();
        camera.getWorldDirection(right);right.cross(camera.up).normalize();
        targetRef.current.add(right.multiplyScalar(-dx*0.5));
        targetRef.current.add(new THREE.Vector3(0,1,0).multiplyScalar(dy*0.3));
        updateCamera();
      }
      // Tooltip
      mouse.x=(e.clientX/W)*2-1; mouse.y=-(e.clientY/H)*2+1;
      raycaster.setFromCamera(mouse,camera);
      let hit=false;
      for(const grp of buildingMeshes){
        if(raycaster.intersectObjects(grp.children,true).length>0){
          setTooltip({x:e.clientX,y:e.clientY,name:grp.userData.name});
          hit=true; break;
        }
      }
      if(!hit) setTooltip(null);
      d.prev={x:e.clientX,y:e.clientY};
    };
    const onUp=()=>{dragRef.current.isDragging=false;dragRef.current.isRight=false;};
    const onWheel=e=>{
      sphericalRef.current.radius=Math.max(100,Math.min(900,sphericalRef.current.radius+e.deltaY*0.5));
      updateCamera();
    };
    const onClick=e=>{
      mouse.x=(e.clientX/W)*2-1;mouse.y=-(e.clientY/H)*2+1;
      raycaster.setFromCamera(mouse,camera);
      // 사람 클릭 체크
      let personHit=false;
      const allPersons=scene.children.filter(c=>c.userData&&c.userData.isPerson);
      for(const person of allPersons){
        if(raycaster.intersectObjects(person.children,true).length>0){
          personHit=true;
          const msg=personDialogs[Math.floor(Math.random()*personDialogs.length)];
          const pos=new THREE.Vector3();
          person.getWorldPosition(pos); pos.y+=8;
          pos.project(camera);
          const sx=(pos.x*0.5+0.5)*W, sy=(-pos.y*0.5+0.5)*H;
          setSpeechBubble({x:sx,y:sy,msg,key:Date.now()});
          if(speechTimerRef.current)clearTimeout(speechTimerRef.current);
          speechTimerRef.current=setTimeout(()=>setSpeechBubble(null),3000);
          break;
        }
      }
      if(personHit) return;
      // 건물 클릭
      for(const grp of buildingMeshes){
        if(raycaster.intersectObjects(grp.children,true).length>0){
          setSelected(grp.userData); break;
        }
      }
    };
    const onCtx=e=>e.preventDefault();

    el.addEventListener("mousedown",onDown);
    el.addEventListener("mousemove",onMove);
    window.addEventListener("mouseup",onUp);
    el.addEventListener("wheel",onWheel);
    el.addEventListener("click",onClick);
    el.addEventListener("contextmenu",onCtx);

    const onResize=()=>{
      const w2=container.clientWidth,h2=container.clientHeight;
      camera.aspect=w2/h2; camera.updateProjectionMatrix();
      renderer.setSize(w2,h2);
    };
    window.addEventListener("resize",onResize);

    return ()=>{
      cancelAnimationFrame(animId);
      el.removeEventListener("mousedown",onDown);
      el.removeEventListener("mousemove",onMove);
      window.removeEventListener("mouseup",onUp);
      el.removeEventListener("wheel",onWheel);
      el.removeEventListener("click",onClick);
      el.removeEventListener("contextmenu",onCtx);
      window.removeEventListener("resize",onResize);
      container.removeChild(renderer.domElement);
      renderer.dispose();
      document.querySelectorAll('.bneck-bubble').forEach(el=>el.remove());
      if(speechTimerRef.current)clearTimeout(speechTimerRef.current);
    };
  }, []);

  return (
    <div style={{width:"100vw",height:"100vh",overflow:"hidden",background:"#0a0a0f",fontFamily:"'Noto Sans KR',sans-serif"}}>
      <div ref={mountRef} style={{width:"100%",height:"100%"}} />

      {/* Title */}
      <div style={{position:"fixed",top:20,left:24,zIndex:100,pointerEvents:"none"}}>
        <h1 style={{fontSize:22,fontWeight:900,letterSpacing:2,color:"#fff",textShadow:"0 2px 20px rgba(0,0,0,0.8)",margin:0}}>
          SK hynix 이천캠퍼스
        </h1>
        <p style={{fontSize:11,color:"#888",letterSpacing:4,textTransform:"uppercase",margin:0}}>3D Campus Visualization</p>
      </div>

      {/* Legend */}
      <div style={{position:"fixed",top:24,right:24,background:"rgba(10,10,20,0.85)",backdropFilter:"blur(20px)",border:"1px solid rgba(255,255,255,0.08)",borderRadius:12,padding:"14px 18px",zIndex:100,fontSize:11}}>
        <div style={{fontSize:10,color:"#666",letterSpacing:2,textTransform:"uppercase",marginBottom:10}}>범례</div>
        {[
          ["#ecc94b","M14 DRAM FAB"],["#48bb78","M16 DRAM FAB"],["#63b3ed","M10 FAB"],
          ["#e07098","DRAM WT (검사동)"],["#c4956a","P&T (패키지·테스트)"],["#44aaee","OHT 연결통로"]
        ].map(([c,l])=>(
          <div key={l} style={{display:"flex",alignItems:"center",marginBottom:6,color:"#aaa"}}>
            <div style={{width:14,height:14,borderRadius:3,marginRight:8,flexShrink:0,background:c}} />{l}
          </div>
        ))}
      </div>

      {/* Bottleneck Test */}
      <div style={{position:"fixed",top:280,right:24,zIndex:100,background:"rgba(10,10,20,0.85)",backdropFilter:"blur(20px)",border:"1px solid rgba(255,255,255,0.08)",borderRadius:12,padding:"14px 18px"}}>
        <div style={{fontSize:10,color:"#666",letterSpacing:2,textTransform:"uppercase",marginBottom:8}}>병목 테스트</div>
        <button onClick={toggleBottleneck} style={{
          width:"100%",padding:"10px 16px",
          background:bottleneck?"linear-gradient(135deg,rgba(255,30,30,0.4),rgba(200,0,0,0.3))":"linear-gradient(135deg,rgba(255,50,30,0.2),rgba(255,100,0,0.15))",
          border:`1px solid ${bottleneck?"rgba(255,50,50,0.6)":"rgba(255,80,40,0.4)"}`,
          borderRadius:8,color:"#ff6644",fontSize:13,fontWeight:700,cursor:"pointer",
          transition:"all 0.3s",fontFamily:"inherit"
        }}>
          {bottleneck?"🔥 병목 해제":"⚠ 병목 테스트"}
        </button>
        <div style={{fontSize:10,color:bottleneck?"#ff4444":"#555",marginTop:6,textAlign:"center"}}>
          {bottleneck?"M14/M16 QUE 병목 경고 중...":"M14/M16 QUE 병목 예측"}
        </div>
      </div>

      {/* Time Toggle */}
      <div style={{position:"fixed",top:75,left:24,zIndex:200,background:"rgba(10,10,20,0.85)",borderRadius:10,border:"1px solid rgba(255,255,255,0.1)",backdropFilter:"blur(8px)",padding:"8px 12px",display:"flex",gap:6}}>
        {[["morning","🌅 아침"],["auto","🕐 실시간"],["night","🌙 밤"]].map(([m,lb])=>(
          <button key={m} onClick={()=>setTimeMode(m)} style={{
            padding:"6px 14px",border:"none",borderRadius:6,cursor:"pointer",fontSize:13,fontWeight:600,
            background:timeMode===m?"rgba(100,200,255,0.3)":"rgba(100,200,255,0.08)",
            color:timeMode===m?"#88ddff":"#556",
            boxShadow:timeMode===m?"0 0 8px rgba(100,200,255,0.3)":"none",
            transition:"all 0.3s"
          }}>{lb}</button>
        ))}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position:"fixed",left:tooltip.x,top:tooltip.y,transform:"translate(-50%,-120%)",
          background:"rgba(0,0,0,0.85)",backdropFilter:"blur(8px)",border:"1px solid rgba(255,255,255,0.15)",
          borderRadius:6,padding:"6px 10px",fontSize:12,color:"#fff",fontWeight:600,zIndex:200,
          pointerEvents:"none",whiteSpace:"nowrap"
        }}>{tooltip.name}</div>
      )}

      {/* Speech Bubble */}
      {speechBubble && (
        <div key={speechBubble.key} style={{
          position:"fixed",left:speechBubble.x,top:speechBubble.y,transform:"translate(-50%,-120%)",
          background:"rgba(255,255,255,0.95)",borderRadius:16,padding:"10px 18px",
          fontSize:13,color:"#222",fontWeight:600,zIndex:400,pointerEvents:"none",
          whiteSpace:"nowrap",boxShadow:"0 4px 20px rgba(0,0,0,0.3)",
          animation:"bubblePop 0.3s ease-out"
        }}>
          {speechBubble.msg}
          <div style={{
            position:"absolute",bottom:-8,left:"50%",transform:"translateX(-50%)",
            width:0,height:0,borderLeft:"8px solid transparent",borderRight:"8px solid transparent",
            borderTop:"8px solid rgba(255,255,255,0.95)"
          }}/>
        </div>
      )}
      <style>{`@keyframes bubblePop{from{transform:translate(-50%,-120%) scale(0.5);opacity:0}to{transform:translate(-50%,-120%) scale(1);opacity:1}}`}</style>

      {/* Info Panel */}
      <div style={{
        position:"fixed",bottom:24,left:24,background:"rgba(10,10,20,0.92)",backdropFilter:"blur(20px)",
        border:"1px solid rgba(255,255,255,0.08)",borderRadius:14,zIndex:100,
        minWidth:280,maxWidth:340,maxHeight:"70vh",overflowY:"auto"
      }}>
        <div style={{padding:"14px 18px 10px",borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
          <div style={{fontSize:17,fontWeight:800,color:"#fff",display:"flex",alignItems:"center",gap:8}}>
            <span style={{display:"inline-block",width:10,height:10,borderRadius:3,
              background:selected?`#${(selected.color||0x4488ff).toString(16).padStart(6,"0")}`:"#4af"}} />
            {selected?selected.name:"SK hynix 이천"}
          </div>
          <div style={{fontSize:10,color:"#888",letterSpacing:2,textTransform:"uppercase"}}>
            {selected?selected.type:"캠퍼스 전경"}
          </div>
        </div>
        <div style={{padding:"12px 18px 16px"}}>
          {selected ? (
            <>
              <div style={{fontSize:12,color:"#aaa",lineHeight:1.6,marginBottom:10,whiteSpace:"pre-line"}}>
                {selected.detail}
              </div>
              {selected.specs && (
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginBottom:12}}>
                  {Object.entries(selected.specs).map(([k,v])=>(
                    <div key={k} style={{background:"rgba(255,255,255,0.03)",borderRadius:6,padding:"7px 10px"}}>
                      <div style={{fontSize:9,color:"#555",letterSpacing:1}}>
                        {{floors:"층수",process:"공정",size:"크기",logistics:"물류",role:"역할",link:"연결"}[k]||k}
                      </div>
                      <div style={{fontSize:13,fontWeight:700,color:"#ddd",marginTop:1}}>{v}</div>
                    </div>
                  ))}
                </div>
              )}
              {selected.processFlow && (
                <>
                  <div style={{fontSize:9,color:"#556",letterSpacing:2,marginBottom:6,paddingBottom:4,borderBottom:"1px solid rgba(255,255,255,0.04)"}}>
                    주요 공정
                  </div>
                  <div style={{display:"flex",flexWrap:"wrap",alignItems:"center",gap:0,marginBottom:10}}>
                    {selected.processFlow.map((s,i)=>(
                      <span key={i}>
                        <span style={{fontSize:10,color:"#bbb",background:"rgba(255,255,255,0.04)",padding:"3px 8px",borderRadius:4,whiteSpace:"nowrap"}}>{s}</span>
                        {i<selected.processFlow.length-1 && <span style={{fontSize:10,color:"#4af",margin:"0 3px"}}>→</span>}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </>
          ) : (
            <div style={{fontSize:12,color:"#aaa",lineHeight:1.6}}>
              건물을 클릭하면<br/>생산 공정 & 이동 정보를 볼 수 있습니다
            </div>
          )}
        </div>
      </div>

      {/* Controls hint */}
      <div style={{
        position:"fixed",bottom:24,right:24,background:"rgba(10,10,20,0.7)",backdropFilter:"blur(12px)",
        border:"1px solid rgba(255,255,255,0.06)",borderRadius:10,padding:"10px 14px",zIndex:100,
        fontSize:10,color:"#666",lineHeight:1.8
      }}>
        <span style={{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:3,padding:"1px 5px",color:"#999"}}>드래그</span> 회전 &nbsp;
        <span style={{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:3,padding:"1px 5px",color:"#999"}}>우클릭</span> 이동 &nbsp;
        <span style={{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:3,padding:"1px 5px",color:"#999"}}>휠</span> 줌
      </div>
    </div>
  );
}
