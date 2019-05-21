pad=40;
initd=200;
x1=0;
y1=0;
color(yellow, .75){
d1=initd;
cube([3,d1,d1]);

translate([pad/2,x1+d1,d1/2]){ rotate([0,0,90]){color("blue") {
    text(">", size=pad);
}}}

d2=d1-2;
x2=d1+pad;
translate([-64,x2,y1]) {
cube([64,d2,d2]);
}

translate([pad/2,x2+d2,d2/2]){ rotate([0,0,90]){color("blue") {
    text(">", size=pad);
}}}

d3=d2-2;
x3=x2+d2+pad;
translate([-64,x3,y1]) {
cube([64,d3,d3]);
}
d4=d3/2;
x4=x3+d3-d4;
y2=y1-d4-pad;
translate([-64,x4,y2]) {
cube([64,d4,d4]);
}
d5=d4-2;
x5=x4+d4+pad;
translate([-128,x5,y2]) {
cube([128,d5,d5]);
}
d6=d5-2;
x6=x5+d5+pad;
translate([-128,x6,y2]) {
cube([128,d6,d6]);
}

d7=d6/2;
x7=x6+d6-d7;
y3=y2-d7-pad;
translate([-128,x7,y3]) {
cube([128,d7,d7]);
}

d8=d7-2;
x8=x7+d7+pad;
translate([-256,x8,y3]) {
cube([256,d8,d8]);
}

d9=d8-2;
x9=x8+d8+pad;
translate([-256,x9,y3]) {
cube([256,d9,d9]);
}

d10=d9/2;
x10=x9+d9-d10;
y4=y3-d10-pad;
translate([-256,x10,y4]) {
cube([256,d10,d10]);
}

d11=d10-2;
x11=x10+d10+pad;
translate([-512,x11,y4]) {
cube([512,d11,d11]);
}

d12=d11-2;
x12=x11+d11+pad;
translate([-512,x12,y4]) {
cube([512,d12,d12]);
}

d13=d12/2;
x13=x12+d12-d13;
y5=y4-d13-pad;
translate([-512,x13,y5]) {
cube([512,d13,d13]);
}

d14=d13-2;
x14=x13+d13+pad;
translate([-1024,x14,y5]) {
cube([1024,d14,d14]);
}

d15=d14-2;
x15=x14+d14+pad;
translate([-1024,x15,y5]) {
cube([1024,d15,d15]);
}

d16=d15*2;
x16=x15;
translate([-1024,x16,y4]) {
cube([1024,d16,d16]);
}

d17=d16-2;
x17=x16+d16+pad;
translate([-512,x17,y4]) {
cube([512,d17,d17]);
}

d18=d17-2;
x18=x17+d17+pad;
translate([-512,x18,y4]) {
cube([512,d18,d18]);
}

d19=d18*2;
x19=x18;
translate([-512,x18,y3]) {
cube([512,d19,d19]);
}

d20=d19-2;
x20=x19+pad;
translate([-256,x20,y3]) {
cube([256,d20,d20]);
}

d21=d20-2;
x21=x20+pad;
translate([-256,x21,y3]) {
cube([256,d21,d21]);
}

d22=d21*2;
x22=x21;
translate([-256,x22,y2]) {
cube([256,d22,d22]);
}

d23=d22-2;
x23=x22+pad;
translate([-128,x23,y2]) {
cube([128,d23,d23]);
}

d24=d23-2;
x24=x23+pad;
translate([-128,x24,y2]) {
cube([128,d24,d24]);
}

d25=d24*2;
x25=x24;
translate([-128,x25,y1]) {
cube([128,d25,d25]);
}

d26=d25-2;
x26=x25+pad;
translate([-64,x26,y1]) {
cube([64,d26,d26]);
}

d27=d26-2;
x27=x26+pad;
translate([-64,x27,y1]) {
cube([64,d27,d27]);
}

d28=d27;
x28=x27+pad;
translate([-3,x28,y1]) {
cube([3,d28,d28]);
}


}