import torch.nn as nn
import torch

#160
class HulNet(nn.Module):
    def __init__(self):
        funit = 32
        super(HulNet, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec1to3std = nn.Sequential(
            # in: 2
            # out: 2
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec3to6str = nn.Sequential(
            # in: 2+2
            # out: 2
            nn.ConvTranspose2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec10to14std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 4+4, out:2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec14to42str = nn.Sequential(
            # in: 4+2, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 2+4, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec50to150str = nn.Sequential(
            # in: 4+2, out: 4
            nn.ConvTranspose2d(6*funit, 4*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.dec150to154std = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+2
            nn.ConvTranspose2d(6*funit, 3, 3),
            nn.ReLU(),
        )


    def forward(self, x):
        # 160 to 150
        dil160 = x#.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158#.clone()
        upc158 = std158#.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154#.clone()
        upc154 = cat154#.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        upc150 = cat150#.clone()
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50#.clone()
        upc50 = str50#.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46#.clone()

        upc46 = cat46#.clone()
        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        upc42 = cat42#.clone()
        str14 = self.enc42to14str(cat42)
        dil14 = str14#.clone()
        upc14 = str14#.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10#.clone()
        upc10 = cat10#.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)
        upc6 = cat6#.clone()

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        upc3 = str3#.clone()
        std1 = self.enc3to1std(str3)
        del(str3)

        # up

        std3 = torch.cat([upc3, self.dec1to3std(std1)], 1)
        del(std1)

        str6 = torch.cat([upc6, self.dec3to6str(std3)], 1)
        del(upc6, std3)
        dil6 = str6#.clone()
        cat10 = torch.cat([upc10, self.dec6to10std(str6), self.dec6to10dil(dil6)], 1)
        del(dil6, str6, upc10)
        dil10 = cat10#.clone()
        cat14 = torch.cat([upc14, self.dec10to14std(cat10), self.dec10to14dil(dil10)], 1)
        del(cat10, dil10, upc14)
        cat42 = torch.cat([upc42, self.dec14to42str(cat14)], 1)
        del(cat14, upc42)
        dil42 = cat42#.clone()
        cat46 = torch.cat([upc46, self.dec42to46std(cat42), self.dec42to46dil(dil42)], 1)
        del(dil42, cat42, upc46)
        dil46 = cat46#.clone()
        cat50 = torch.cat([upc50, self.dec46to50std(cat46), self.dec46to50dil(dil46)], 1)
        del(cat46, dil46, upc50)
        cat150 = torch.cat([upc150, self.dec50to150str(cat50)], 1)
        del(upc150, cat50)
        dil150 = cat150#.clone()
        cat154 = torch.cat([upc154, self.dec150to154std(cat150), self.dec150to154dil(dil150)], 1)
        del(cat150, dil150, upc154)
        dil154 = cat154#.clone()
        cat158 = torch.cat([upc158, self.dec154to158std(cat154), self.dec154to158dil(dil154)], 1)
        del(dil154, upc158, cat154)
        return self.dec158to160std(cat158)


#144
class Hul144Disc(nn.Module):
    def __init__(self, input_channels = 3, funit = 32, finalpool = False):
        super(Hul144Disc, self).__init__()
        self.funit = funit
        self.enc144to142std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.enc142to138std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc138to134std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc142to138dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc138to134dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc144to134dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc134to132std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc132to44str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc44to40std = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc40to36std = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.enc44to40dil = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc40to36dil = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc36to12str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.enc12to8std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc8to4std = nn.Sequential(
            nn.Conv2d(12*funit, 6*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.enc12to8dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc8to4dil = nn.Sequential(
            nn.Conv2d(12*funit, 6*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        if not finalpool:
            self.enc4to2std = nn.Sequential(
                nn.Conv2d(12*funit, 4*funit, 3, bias=False),
                #nn.BatchNorm2d(2*funit),
                nn.PReLU(),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(4*funit, 1, 2),
                nn.Sigmoid()
            )
        else:
            self.enc4to2std = nn.Sequential(
                nn.Conv2d(12*funit, 1, 3, bias=False),
                #nn.BatchNorm2d(2*funit),
                nn.Sigmoid(),
            )
            self.decide = nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
            )
    def forward(self, x):
        lay_i = self.enc144to142std(x)
        lay_i = torch.cat([self.enc142to138std(lay_i), self.enc142to138dil(lay_i)], 1)
        layer = torch.cat([self.enc138to134std(lay_i), self.enc138to134dil(lay_i), self.enc144to134dil(x)], 1)
        del(x, lay_i)
        layer = self.enc134to132std(layer)
        layer = self.enc132to44str(layer)
        layer = torch.cat([self.enc44to40std(layer), self.enc44to40dil(layer)], 1)
        layer = torch.cat([self.enc40to36std(layer), self.enc40to36dil(layer)], 1)
        layer = self.enc36to12str(layer)
        layer = torch.cat([self.enc12to8std(layer), self.enc12to8dil(layer)], 1)
        layer = torch.cat([self.enc8to4std(layer), self.enc8to4dil(layer)], 1)
        layer = self.enc4to2std(layer)
        return self.decide(layer)
