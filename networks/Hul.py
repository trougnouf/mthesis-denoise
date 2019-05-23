import torch.nn as nn
import torch

#160
class Hul160Net(nn.Module):
    def __init__(self):
        funit = 32
        super(Hul160Net, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec1to3std = nn.Sequential(
            # in: 2
            # out: 2
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec3to6str = nn.Sequential(
            # in: 2+2
            # out: 2
            nn.ConvTranspose2d(4*funit, 2*funit, 2, stride=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )

        self.dec6to10std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec10to14std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 4+4, out:2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec14to42str = nn.Sequential(
            # in: 4+2, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec42to46std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 2+4, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec46to50std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec50to150str = nn.Sequential(
            # in: 4+2, out: 4
            nn.ConvTranspose2d(6*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.dec150to154std = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+2
            nn.ConvTranspose2d(6*funit, 3, 3),
            nn.ReLU(),
        )


    def forward(self, x):
        # down
        # 160 to 150
        l158 = self.enc160to158std(x)
        l154 = torch.cat([self.enc158to154std(l158), self.enc158to154dil(l158)], 1)
        l150 = torch.cat([self.enc154to150std(l154), self.enc154to150dil(l154), self.enc160to150dil(x)], 1)
        del(x)
        l50 = self.enc150to50str(l150)
        l46 = torch.cat([self.enc50to46std(l50), self.enc50to46dil(l50)], 1)
        l42 = torch.cat([self.enc46to42std(l46), self.enc46to42dil(l46)], 1)
        l14 = self.enc42to14str(l42)
        l10 = torch.cat([self.enc14to10std(l14), self.enc14to10dil(l14)], 1)
        l6 = torch.cat([self.enc10to6std(l10), self.enc10to6dil(l10)], 1)
        l3 = self.enc6to3str(l6)    # k2s2
        l1 = self.enc3to1std(l3)
        # up
        l3 = torch.cat([l3, self.dec1to3std(l1)], 1)
        del(l1)

        l6 = torch.cat([l6, self.dec3to6str(l3)], 1)
        del(l3)
        l10 = torch.cat([l10, self.dec6to10std(l6), self.dec6to10dil(l6)], 1)
        del(l6)
        l14 = torch.cat([l14, self.dec10to14std(l10), self.dec10to14dil(l10)], 1)
        del(l10)
        l42 = torch.cat([l42, self.dec14to42str(l14)], 1)
        l46 = torch.cat([l46, self.dec42to46std(l42), self.dec42to46dil(l42)], 1)
        del(l42)
        l50 = torch.cat([l50, self.dec46to50std(l46), self.dec46to50dil(l46)], 1)
        del(l46)
        l150 = torch.cat([l150, self.dec50to150str(l50)], 1)
        del(l50)
        l154 = torch.cat([l154, self.dec150to154std(l150), self.dec150to154dil(l150)], 1)
        del(l150)
        l158 = torch.cat([l158, self.dec154to158std(l154), self.dec154to158dil(l154)], 1)
        del(l154)
        return self.dec158to160std(l158)

# single-net: BS = 17 w/ 8GB GPU, 27 w/ 11GB GPU
class Hul128Net(nn.Module):
    def __init__(self):
        funit = 32
        super(Hul128Net, self).__init__()
        self.enc128to126std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc126to122std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc122to118std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc126to122dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc122to118dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc128to118dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc118to114std = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc118to114dil = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc114to38str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc38to34std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc38to34dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc30to10str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )

        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc6to2std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc6to2dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.dec2to6std = nn.Sequential(
            # in 0+12 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec6to10std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec2to6dil = nn.Sequential(
            # in: 0+8 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec6to10dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec10to30str = nn.Sequential(
            # in: 4+6, out:5
            nn.ConvTranspose2d(10*funit, 5*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(5*funit),
        )
        self.dec30to34std = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec30to34dil = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec34to38std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec34to38dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec38to114str = nn.Sequential(
            # in: 4+6, out: 4
            nn.ConvTranspose2d(10*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.dec114to118std = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec114to118dil = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec118to122std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec118to122dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec122to126std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec122to126dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec126to128std = nn.Sequential(
            # in: 2+6
            nn.ConvTranspose2d(8*funit, 3, 3),
            nn.ReLU(),
        )

    def forward(self, x):
        # down
        # 160 to 150
        l126 = self.enc128to126std(x)
        l122 = torch.cat([self.enc126to122std(l126), self.enc126to122dil(l126)], 1)
        l118 = torch.cat([self.enc122to118std(l122), self.enc122to118dil(l122), self.enc128to118dil(x)], 1)
        del(x)
        l114 = torch.cat([self.enc118to114std(l118), self.enc118to114dil(l118)], 1)
        l38 = self.enc114to38str(l114)
        l34 = torch.cat([self.enc38to34std(l38), self.enc38to34dil(l38)], 1)
        l30 = torch.cat([self.enc34to30std(l34), self.enc34to30dil(l34)], 1)
        l10 = self.enc30to10str(l30)
        l6 = torch.cat([self.enc10to6std(l10), self.enc10to6dil(l10)], 1)
        l2 = torch.cat([self.enc6to2std(l6), self.enc6to2dil(l6)], 1)
        # up
        l6 = torch.cat([l6, self.dec2to6std(l2), self.dec2to6dil(l2)], 1)
        del(l2)
        l10 = torch.cat([l10, self.dec6to10std(l6), self.dec6to10dil(l6)], 1)
        del(l6)
        l30 = torch.cat([l30, self.dec10to30str(l10)], 1)
        del(l10)
        l34 = torch.cat([l34, self.dec30to34std(l30), self.dec30to34dil(l30)], 1)
        del(l30)
        l38 = torch.cat([l38, self.dec34to38std(l34), self.dec34to38dil(l34)], 1)
        del(l34)
        l114 = torch.cat([l114, self.dec38to114str(l38)], 1)
        del(l38)
        l118 = torch.cat([l118, self.dec114to118std(l114), self.dec114to118dil(l114)], 1)
        del(l114)
        l122 = torch.cat([l122, self.dec118to122std(l118), self.dec118to122dil(l118)], 1)
        del(l118)
        l126 = torch.cat([l126, self.dec122to126std(l122), self.dec122to126dil(l122)], 1)
        return self.dec126to128std(l126)

# single-net: BS = 17 w/ 8GB GPU, 27 w/ 11GB GPU
class Hulb128Net(nn.Module):
    def __init__(self, funit=32, activation='PReLU'):
        super(Hulb128Net, self).__init__()
        self.enc128to126std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc126to122std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc122to118std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc126to122dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc122to118dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc128to118dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc118to114std = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc118to114dil = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc114to38str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc38to34std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc38to34dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc30to10str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )

        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc6to2std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc6to2dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.dec2to6std = nn.Sequential(
            # in 0+12 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec6to10std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec2to6dil = nn.Sequential(
            # in: 0+8 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec6to10dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec10to30str = nn.Sequential(
            # in: 4+6, out:5
            nn.ConvTranspose2d(10*funit, 5*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(5*funit),
        )
        self.dec30to34std = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec30to34dil = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec34to38std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec34to38dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec38to114str = nn.Sequential(
            # in: 4+6, out: 4
            nn.ConvTranspose2d(10*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.dec114to118std = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec114to118dil = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec118to122std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec118to122dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec122to126std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec122to126dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec126to128std = nn.Sequential(
            # in: 2+6
            nn.ConvTranspose2d(8*funit, 2*funit, 3),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(2*funit, 3, 1),
        )
        if activation is None or activation == 'None':
            self.activation = None
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU(init=0.01)
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        # down
        # 160 to 150
        l126 = self.enc128to126std(x)
        l122 = torch.cat([self.enc126to122std(l126), self.enc126to122dil(l126)], 1)
        l118 = torch.cat([self.enc122to118std(l122), self.enc122to118dil(l122), self.enc128to118dil(x)], 1)
        del(x)
        l114 = torch.cat([self.enc118to114std(l118), self.enc118to114dil(l118)], 1)
        l38 = self.enc114to38str(l114)
        l34 = torch.cat([self.enc38to34std(l38), self.enc38to34dil(l38)], 1)
        l30 = torch.cat([self.enc34to30std(l34), self.enc34to30dil(l34)], 1)
        l10 = self.enc30to10str(l30)
        l6 = torch.cat([self.enc10to6std(l10), self.enc10to6dil(l10)], 1)
        l2 = torch.cat([self.enc6to2std(l6), self.enc6to2dil(l6)], 1)
        # up
        l6 = torch.cat([l6, self.dec2to6std(l2), self.dec2to6dil(l2)], 1)
        del(l2)
        l10 = torch.cat([l10, self.dec6to10std(l6), self.dec6to10dil(l6)], 1)
        del(l6)
        l30 = torch.cat([l30, self.dec10to30str(l10)], 1)
        del(l10)
        l34 = torch.cat([l34, self.dec30to34std(l30), self.dec30to34dil(l30)], 1)
        del(l30)
        l38 = torch.cat([l38, self.dec34to38std(l34), self.dec34to38dil(l34)], 1)
        del(l34)
        l114 = torch.cat([l114, self.dec38to114str(l38)], 1)
        del(l38)
        l118 = torch.cat([l118, self.dec114to118std(l114), self.dec114to118dil(l114)], 1)
        del(l114)
        l122 = torch.cat([l122, self.dec118to122std(l118), self.dec118to122dil(l118)], 1)
        del(l118)
        l126 = torch.cat([l126, self.dec122to126std(l122), self.dec122to126dil(l122)], 1)
        res = self.dec126to128std(l126)
        if self.activation is None:
            return res
        else:
            return self.activation(res)


#144
class Hul144Disc(nn.Module):
    def __init__(self, input_channels = 3, funit = 32, finalpool = False):
        super(Hul144Disc, self).__init__()
        self.funit = funit
        self.enc144to142std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc142to138std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc138to134std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc142to138dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc138to134dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc144to134dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc134to132std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc132to44str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc44to40std = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc40to36std = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )

        self.enc44to40dil = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc40to36dil = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc36to12str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )

        self.enc12to8std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc8to4std = nn.Sequential(
            nn.Conv2d(12*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )

        self.enc12to8dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc8to4dil = nn.Sequential(
            nn.Conv2d(12*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        if not finalpool:
            self.enc4to2std = nn.Sequential(
                nn.Conv2d(12*funit, 4*funit, 3, bias=True),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(4*funit, 1, 2),
                nn.Sigmoid()
            )
        else:
            self.enc4to2std = nn.Sequential(
                nn.Conv2d(12*funit, 1, 3, bias=True),
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

#112
# w/ Hul128Net BS 12 on 7GB GPU, 20 on 11GB GPU or 19 if conditional
class Hul112Disc(nn.Module):
    def __init__(self, input_channels = 3, funit = 32, finalpool = False, out_activation = 'PReLU'):
        super(Hul112Disc, self).__init__()
        self.funit = funit
        self.enc112to108std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc108to104std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc112to108dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=2),
            nn.PReLU(init=0.01),
        )
        self.enc108to104dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc104to102std = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc112to102dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc102to34str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc30to26std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc26to22std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc22to18std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )

        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc30to26dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc26to22dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc22to18dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )

        self.enc18to6str = nn.Sequential(
            nn.Conv2d(8*funit, 8*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(8*funit),
        )


        if not finalpool:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.BatchNorm2d(6*funit),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 1*funit, 2),
                nn.PReLU(init=0.01),
                nn.Conv2d(1*funit, 1, 1),
                #nn.Sigmoid()
            )
        else:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.BatchNorm2d(6*funit),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 2*funit, 1),
                nn.PReLU(init=0.01),
                nn.Conv2d(2*funit, 1, 1),
                #nn.Sigmoid(),
                nn.AdaptiveMaxPool2d(1),
            )
        if out_activation is None or out_activation == 'None':
            self.out_activation = None
        elif out_activation == 'PReLU':
            self.out_activation = nn.PReLU(init=0.01)
        elif out_activation == 'Sigmoid':
            self.out_activation = nn.Sigmoid()
    def forward(self, x):
        layer = torch.cat([self.enc112to108std(x), self.enc112to108dil(x)], 1)
        layer = torch.cat([self.enc108to104std(layer), self.enc108to104dil(layer)], 1)
        layer = torch.cat([self.enc104to102std(layer), self.enc112to102dil(x)], 1)
        layer = self.enc102to34str(layer)
        layer = torch.cat([self.enc34to30std(layer), self.enc34to30dil(layer)], 1)
        layer = torch.cat([self.enc30to26std(layer), self.enc30to26dil(layer)], 1)
        layer = torch.cat([self.enc26to22std(layer), self.enc26to22dil(layer)], 1)
        layer = torch.cat([self.enc22to18std(layer), self.enc22to18dil(layer)], 1)
        layer = self.enc18to6str(layer)
        layer = torch.cat([self.enc6to2std(layer), self.enc6to2dil(layer)], 1)
        layer = self.decide(layer)
        if self.out_activation is not None:
            return self.out_activation(layer)
        return layer


HulNet = Hul160Net  # compatibility

# testing
# single-net: BS = 17 w/ 8GB GPU, 27 w/ 11GB GPU
class HulbNoBN128Net(nn.Module):
    def __init__(self, funit=32, activation='PReLU'):
        super(HulbNoBN128Net, self).__init__()
        self.enc128to126std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc126to122std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc122to118std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc126to122dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc122to118dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc128to118dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc118to114std = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc118to114dil = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc114to38str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc38to34std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc38to34dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc30to10str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )

        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc6to2std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc6to2dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec2to6std = nn.Sequential(
            # in 0+12 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec6to10std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec2to6dil = nn.Sequential(
            # in: 0+8 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec6to10dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec10to30str = nn.Sequential(
            # in: 4+6, out:5
            nn.ConvTranspose2d(10*funit, 5*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec30to34std = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec30to34dil = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec34to38std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec34to38dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec38to114str = nn.Sequential(
            # in: 4+6, out: 4
            nn.ConvTranspose2d(10*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec114to118std = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec114to118dil = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec118to122std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec118to122dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec122to126std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec122to126dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec126to128std = nn.Sequential(
            # in: 2+6
            nn.ConvTranspose2d(8*funit, 2*funit, 3),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(2*funit, 3, 1),
        )
        if activation == None or activation == 'None':
            self.activation = None
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU(init=0.01)
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
    def forward(self, x):
        # down
        # 160 to 150
        l126 = self.enc128to126std(x)
        l122 = torch.cat([self.enc126to122std(l126), self.enc126to122dil(l126)], 1)
        l118 = torch.cat([self.enc122to118std(l122), self.enc122to118dil(l122), self.enc128to118dil(x)], 1)
        del(x)
        l114 = torch.cat([self.enc118to114std(l118), self.enc118to114dil(l118)], 1)
        l38 = self.enc114to38str(l114)
        l34 = torch.cat([self.enc38to34std(l38), self.enc38to34dil(l38)], 1)
        l30 = torch.cat([self.enc34to30std(l34), self.enc34to30dil(l34)], 1)
        l10 = self.enc30to10str(l30)
        l6 = torch.cat([self.enc10to6std(l10), self.enc10to6dil(l10)], 1)
        l2 = torch.cat([self.enc6to2std(l6), self.enc6to2dil(l6)], 1)
        # up
        l6 = torch.cat([l6, self.dec2to6std(l2), self.dec2to6dil(l2)], 1)
        del(l2)
        l10 = torch.cat([l10, self.dec6to10std(l6), self.dec6to10dil(l6)], 1)
        del(l6)
        l30 = torch.cat([l30, self.dec10to30str(l10)], 1)
        del(l10)
        l34 = torch.cat([l34, self.dec30to34std(l30), self.dec30to34dil(l30)], 1)
        del(l30)
        l38 = torch.cat([l38, self.dec34to38std(l34), self.dec34to38dil(l34)], 1)
        del(l34)
        l114 = torch.cat([l114, self.dec38to114str(l38)], 1)
        del(l38)
        l118 = torch.cat([l118, self.dec114to118std(l114), self.dec114to118dil(l114)], 1)
        del(l114)
        l122 = torch.cat([l122, self.dec118to122std(l118), self.dec118to122dil(l118)], 1)
        del(l118)
        l126 = torch.cat([l126, self.dec122to126std(l122), self.dec122to126dil(l122)], 1)
        res = self.dec126to128std(l126)
        if self.activation is None:
            return res
        else:
            return self.activation(res)


