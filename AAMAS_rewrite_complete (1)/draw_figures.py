from pathlib import Path
import re, math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("FigureSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("FigureSansBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent
class Drawing:
 def __init__(self,name,w,h):
  self.w=w; self.h=h; self.c=canvas.Canvas(str(ROOT/'figures'/f'{name}.pdf'),pagesize=(w,h)); self.c.setTitle(name.replace('_',' ')); self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']; self.name=name
 def line(self,x1,y1,x2,y2,color='#334155',width=1.7,dash=False,arrow=False):
  c=self.c;c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.setDash(5,4) if dash else c.setDash();c.line(x1,self.h-y1,x2,self.h-y2)
  self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
  if arrow:
   a=math.atan2(y2-y1,x2-x1);pts=[(x2,y2),(x2-7*math.cos(a-.45),y2-7*math.sin(a-.45)),(x2-7*math.cos(a+.45),y2-7*math.sin(a+.45))];p=c.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1]);[p.lineTo(x,self.h-y) for x,y in pts[1:]];p.close();c.setFillColor(HexColor(color));c.drawPath(p,fill=1,stroke=0);self.parts.append(f'<polygon points="'+ ' '.join(f'{x},{y}' for x,y in pts)+f'" fill="{color}"/>')
 def box(self,x,y,w,h,fill='#f8fafc',stroke='#cbd5e1',r=8):
  c=self.c;c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setDash();c.setLineWidth(1);c.roundRect(x,self.h-y-h,w,h,r,fill=1,stroke=1);self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
 def txt(self,x,y,s,size=12,bold=False,color='#0f172a',anchor='start'):
  c=self.c;c.setFillColor(HexColor(color));c.setFont('FigureSansBold' if bold else 'FigureSans',size);fn={'start':c.drawString,'middle':c.drawCentredString,'end':c.drawRightString}[anchor];fn(x,self.h-y,s);self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="'+('700' if bold else '400')+f'" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')
 def circle(self,x,y,r,fill,stroke='#334155'):
  c=self.c;c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.circle(x,self.h-y,r,fill=1,stroke=1);self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>')
 def save(self):
  self.c.showPage();self.c.save();(ROOT/'figures'/f'{self.name}.svg').write_text('\n'.join(self.parts+['</svg>']))
# Figure 1: topology, independently drawn, no illustrative generated image.
d=Drawing('system_model',760,430)
d.txt(24,28,'Programmable MA-RIS-ISAC environment',19,True)
d.txt(24,49,'Three actuator controllers share one evolving propagation environment',11,color='#475569')
d.box(24,84,187,231,'#eff6ff','#93c5fd');d.txt(117,107,'Base station',14,True,anchor='middle');d.txt(117,126,'M movable antennas',11,anchor='middle')
for i in range(4):
 for j in range(4):
  x=58+35*j;y=146+30*i;d.box(x,y,26,24,'#ffffff','#bfdbfe',2);d.circle(x+13+(-3 if i%2 else 3),y+12,3.8,'#2563eb','#2563eb')
d.txt(117,282,'Cell bounds + minimum spacing',10,anchor='middle');d.txt(117,299,'Power p | Layout P',12,True,anchor='middle')
d.box(310,83,158,114,'#f0fdfa','#5eead4');d.txt(389,104,'RIS: L elements',13,True,anchor='middle')
for i in range(3):
 for j in range(6):d.box(328+j*20,115+i*19,13,13,'#14b8a6','#0f766e',1)
d.txt(389,186,'Phase vector theta',11,anchor='middle')
d.box(559,91,177,119,'#f8fafc','#94a3b8');d.txt(647,112,'K mobile users',13,True,anchor='middle')
for j in range(3):
 x=594+j*49;d.circle(x,139,6,'#64748b','#64748b');d.box(x-9,148,18,24,'#e2e8f0','#64748b',3)
d.txt(647,194,'Realized rates on true H',10,anchor='middle')
d.box(559,247,177,90,'#fff7ed','#fdba74');d.txt(647,269,'Q sensing directions',13,True,anchor='middle')
for j in range(3):
 x=599+j*48;d.circle(x,298,12,'#fff7ed','#ea580c');d.circle(x,298,5,'#fed7aa','#ea580c')
d.line(211,157,310,139,'#0f766e',2,arrow=True);d.txt(245,132,'BS-RIS G',10,anchor='middle',color='#0f766e')
d.line(468,141,559,141,'#0f766e',2,arrow=True);d.txt(513,127,'RIS-user g',10,anchor='middle',color='#0f766e')
d.line(211,216,535,216,'#2563eb',2);d.line(535,216,559,185,'#2563eb',2,arrow=True);d.txt(370,234,'Direct BS-user d',11,anchor='middle',color='#2563eb')
d.line(211,280,559,297,'#ea580c',1.8,dash=True,arrow=True);d.txt(357,279,'Direct illumination',10,anchor='middle',color='#c2410c')
d.line(468,188,559,259,'#ea580c',1.8,dash=True,arrow=True);d.txt(490,249,'Reflected',10,anchor='middle',color='#c2410c')
d.box(24,357,712,52,'#f8fafc','#cbd5e1');d.txt(40,378,'Temporal dynamics',11,True);d.txt(178,378,'User mobility + correlated fading; propagation regime fixed within an episode',10)
d.txt(40,396,'Control',11,True);d.txt(178,396,'Power, RIS phases and antenna positions; post-action CSI estimated for low-level precoding',10)
d.save()
# Figure 2: precise information boundary.
d=Drawing('ctde_architecture',760,580)
d.txt(24,27,'Information flow and decentralized execution',19,True)
d.box(24,47,712,82,'#eff6ff','#93c5fd');d.txt(380,69,'Physical environment',14,True,anchor='middle');d.txt(380,88,'Post-action true H -> current estimate H-hat -> BD and power profile -> realized SINR',11,anchor='middle');d.txt(380,108,'True H evaluates interference and sensing; actors receive no true-state input',10,anchor='middle',color='#475569')
for x,name,obs,act in [(24,'BS agent','Delayed aggregate-channel summary','Power profile'),(267,'RIS agent','Delayed reflected-channel summary','Grouped phase increment'),(510,'MA agent','Delayed direct-channel summary','Layout increment')]:
 mid=x+113;d.line(mid,129,mid,159,'#2563eb',1.7,arrow=True);d.txt(mid,150,'Local observation + history',10,anchor='middle',color='#2563eb');d.box(x,162,226,118,'#f8fafc','#94a3b8');d.txt(mid,184,name,14,True,anchor='middle');d.txt(mid,204,obs,10,anchor='middle');d.txt(mid,224,'Encoder -> GRU -> two action heads',10,anchor='middle');d.txt(mid,244,act+' + destination',10,anchor='middle');d.txt(mid,263,'Previous action + delayed inbox',10,anchor='middle',color='#475569')
d.box(24,316,712,65,'#f0fdfa','#5eead4');d.txt(380,337,'Communication router',13,True,anchor='middle');d.txt(380,356,'Local-estimate messages -> directed graph mask -> budget B -> queue for t+1',11,anchor='middle');d.txt(380,372,'Log requests / delivery / drops; charge delivered messages in the shared team reward',10,anchor='middle')
for mid in [137,380,623]:
 d.line(mid,280,mid,316,'#0f766e',1.5,dash=True,arrow=True)
# delayed return around outer margin, no confusion with true-state flow
d.line(14,348,4,348,'#0f766e',1.5,dash=True);d.line(4,348,4,215,'#0f766e',1.5,dash=True);d.line(4,215,24,215,'#0f766e',1.5,dash=True,arrow=True)
d.line(746,348,756,348,'#0f766e',1.5,dash=True);d.line(756,348,756,215,'#0f766e',1.5,dash=True);d.line(756,215,736,215,'#0f766e',1.5,dash=True,arrow=True)
d.line(501,316,501,215,'#0f766e',1.5,dash=True);d.line(501,215,493,215,'#0f766e',1.5,dash=True,arrow=True)
d.txt(422,309,'Requests and semantic messages',10,anchor='middle',color='#0f766e')
d.box(24,404,337,63,'#fff7ed','#fdba74');d.txt(192,426,'Feasibility shield and actuator scheduler',12,True,anchor='middle');d.txt(192,444,'Repair spacing / bounds; wrap phases;',10,anchor='middle');d.txt(192,458,'normalize power; enforce MA update period',10,anchor='middle')
d.box(386,404,350,63,'#f5f3ff','#c4b5fd');d.txt(561,426,'Training only: centralized recurrent critic',12,True,anchor='middle');d.txt(561,444,'True global state + all past actions -> V(s)',10,anchor='middle');d.txt(561,458,'Shared reward -> masked GAE -> PPO updates',10,anchor='middle')
# Action route to shield then return environment, separately on far left within margin

for mid in [137,380,623]:
 d.line(mid+85,280,mid+85,292,'#334155',1.5)
d.line(222,292,708,292,'#334155',1.5);d.line(222,292,20,292,'#334155',1.5);d.line(20,292,20,436,'#334155',1.5);d.line(20,436,24,436,'#334155',1.5,arrow=True)
d.txt(70,308,'Physical actions',9,color='#334155')

d.line(192,467,192,482,'#334155',1.5);d.line(192,482,748,482,'#334155',1.5);d.line(748,482,748,77,'#334155',1.5);d.line(748,77,736,77,'#334155',1.5,arrow=True)
d.txt(385,497,'Shielded physical actions feed the next post-action channel evaluation',10,anchor='middle')
# critic receives true state using purple dotted right inner route
# labelled input not crossing actor boxes
d.txt(561,394,'Global state supplied directly by simulator',10,anchor='middle',color='#7c3aed')
d.box(24,516,712,43,'#ffffff','#cbd5e1');d.txt(40,534,'Blue: delayed local observations',10,color='#2563eb');d.txt(278,534,'Dashed teal: delayed messages',10,color='#0f766e');d.txt(513,534,'Purple box: training-only state',10,color='#7c3aed');d.txt(40,549,'Deployment executes actors, router, shield and low-level precoder; the critic is removed.',10)
d.save()
