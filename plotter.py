from math import sqrt, degrees
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Arc, Polygon, Circle, FancyArrow
import matplotlib.ticker as plticker
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import hashlib
from sympy.functions.elementary.trigonometric import atan2
from scipy.ndimage import rotate


FORCE_COLORS = {
    'red': '#800000',
    'green': '#00CC99',
    'blue': '#31859B',
}


def resolve_force_color(color: str) -> str:
    if color not in FORCE_COLORS:
        raise ValueError("Force color must be one of: 'red', 'green', 'blue'")
    return FORCE_COLORS[color]


class Point():
    def __init__(self, x: float, y:float ,label: str='', labelpos:tuple[str,str]=('top', 'center'),z:float = 0):
        self.x = x
        self.y = y
        self.z = z
        self.label = label
        self.labelx = labelpos[1]
        self.labely = labelpos[0]

    def __repr__(self):
        return f'Point {self.label}: x={self.x}, y={self.y}, z={self.z}'
    
    def angle(self, point):
        return degrees(atan2(self.y - point.y, self.x - point.x))
    
    def distance_to(self, point):
        return sqrt((self.x - point.x)**2 + (self.y - point.y)**2)
        

class Beam():
    def __init__(self, begin: Point, end: Point, label:str=None, labelpos:tuple[str,str]=('top', 'center'), anglelabel:bool=False, anglelabelflip:bool=False):
        self.begin = begin
        self.x1 = begin.x
        self.y1 = begin.y
        self.end = end
        self.x2 = end.x
        self.y2 = end.y
        self.label = label
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.anglelabel =anglelabel
        self.labelflip = anglelabelflip
        self.hinges = []
        
    @property
    def length(self):
        return sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def angle(self):
        return degrees(atan2(self.y2 - self.y1, self.x2 - self.x1))  
    
    def add_hinge(self, is_begin: bool = True):
        if is_begin:
            self.hinges.append((self.begin, is_begin))
        else:
            self.hinges.append((self.end, is_begin))

class Support():
    def __init__(self, point: Point, support_type: str = 'fixed', angle: float = 0.0):
        self.loc = point
        self.x = point.x
        self.y = point.y
        self.set_type(support_type)
        self.angle = angle

    def set_type(self, support_type):
        if support_type not in ['fixed', 'roller', 'pinned']:
            raise ValueError("Use either 'pinned', 'roller' or 'fixed' as support type")
        else:
            self._type = support_type

class RotationSpring():
    def __init__(self, point: Point, value: float=None, unit:str='kNm/rad', labelpos:tuple[str,str]=('top', 'center'),alternative_label:str=''):
        self.x = point.x
        self.y = point.y
        self.value = value
        self.unit = unit
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.alt_label = alternative_label

class TranslationSpring():
    def __init__(self, point: Point, value: float=None, unit:str='kN/m', angle: float=90, labelpos:tuple[str,str]=('top', 'center'),alternative_label:str=''):
        self.x = point.x
        self.y = point.y
        self.value = value
        self.unit = unit
        self.angle = angle
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.alt_label = alternative_label

class PointLoad():
    def __init__(self, point: Point, value: float, unit: str = 'kN', dxdy:tuple[float,float]=(0,1), anglelabel=False, anglelabelflip:bool=False, labelpos:tuple[str,str]=('top', 'center'), alternative_label: str = None, color: str = 'red'):
        self.value = value
        self.dx = dxdy[0]
        self.dy = dxdy[1]
        self.unit = unit
        self.x = point.x
        self.y = point.y
        self.anglelabel = anglelabel
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.labelflip = anglelabelflip
        self.alt_label = alternative_label
        self.color = resolve_force_color(color)


class DistributedLoad():
    def __init__(self, begin_point: Point, end_point: Point, begin_value: float, end_value: float = None, unit: str = 'kN/m', angle: float=90, n_arrow = 6, labelpos:tuple[str,str]=('top', 'center'), labelpos_end:tuple[str,str]=None, alternative_label_begin: str = None, alternative_label_end: str = None, color: str = 'red'):
        self.begin_value = begin_value
        self.end_value = end_value if end_value is not None else begin_value
        self.unit = unit
        self.begin = begin_point
        self.end = end_point
        self.angle = angle
        self.n_arrow = n_arrow
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        if labelpos_end is not None:
            self.labelx_end = labelpos_end[1]
            self.labely_end = labelpos_end[0]
        else:
            self.labelx_end = labelpos[1]
            self.labely_end = labelpos[0]
        self.alt_label_begin = alternative_label_begin
        self.alt_label_end = alternative_label_end
        self.color = resolve_force_color(color)

class Moment():
    def __init__(self, point: Point, value: float=None, unit: str = 'kNm', clock_wise: bool = True, angle: float = 0.0, labelpos:tuple[str,str]=('top', 'center'), alternative_label: str = ''):
        if value is not None and value < 0:
            self.value = -value
            self.clock_wise = not clock_wise
        else:
            self.value = value
            self.clock_wise = clock_wise 
        self.point = point
        self.unit = unit
        self.angle = angle
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.alt_label = alternative_label
        self.color = resolve_force_color(color)

class Length():
    def __init__(self, point1: Point, point2: Point, ax: str='x', xpos: str = 'bottom', ypos: str ='left', alternative_label:str = None):
        self.point1 = point1
        self.point2 = point2
        self.pos = ax
        self.xpos = xpos
        self.ypos = ypos
        self.altlabel =alternative_label


class Structure():
    def __init__(self):
        self._points = set()
        self._beams = set()
        self._pointloads = set()
        self._moments = set()
        self._distributedloads = set()
        self._hinges = set()
        self._fixedsupports = set()
        self._pinnedsupports = set()
        self._rollersupports = set()
        self._rotationsprings = set()
        self._translationsprings = set()
        self._lengths = set()

    def add_point(self, *points):
        for point in points:
            self._points.add(point)

    def add_hinge(self, *points):
        for point in points:
            self._hinges.add(point)
            self._points.add(point)

    def add_beam(self, *beams):
        for beam in beams:
            self._beams.add(beam)
            self.add_point(beam.begin)
            self.add_point(beam.end)

    def add_support(self, *supports):
        for support in supports:
            self.add_point(support.loc)
            if support._type == 'fixed':
                self._fixedsupports.add(support)
            if support._type == 'roller':
                self._rollersupports.add(support)
            if support._type == 'pinned':
                self._pinnedsupports.add(support)

    def add_pointload(self, *pointloads):
        for pointload in pointloads:
            self._pointloads.add(pointload)

    def add_moment(self, *moments):
        for moment in moments:
            self._moments.add(moment)

    def add_distributedload(self, *distributedloads):
        for distributedload in distributedloads:
            self._distributedloads.add(distributedload)

    def add_rotationspring(self, *rotationsprings):
        for rotationspring in rotationsprings:
            self._rotationsprings.add(rotationspring)

    def add_translationspring(self, *translationsprings):
        for translationspring in translationsprings:
            self._translationsprings.add(translationspring)

    def add_length(self, *lengths: Length):
        for length in lengths:
            self._lengths.add(length)

    def xminmax(self):
        xmin = 999
        xmax = -999
        for p in self._points:
            xmin = min(p.x, xmin)
            xmax = max(p.x, xmax)
        return xmin, xmax

    def yminmax(self):
        ymin = 999
        ymax = -999
        for p in self._points:
            ymin = min(p.y, ymin)
            ymax = max(p.y, ymax)
        return ymin, ymax


def plot(structure, seed=None):
    plt.plot([0,0],[0,0],color='black',linewidth=2)
    axs = plt.gca()
    axs.axis('equal')
    axs.axis('off')
    xmin, xmax = structure.xminmax()
    xlength = max(1, xmax - xmin)
    ymin, ymax = structure.yminmax()
    ylength = max(1, ymax - ymin)
    print(xmin, xmax, ymin, ymax)
    # scaler = (np.sqrt(xlength*ylength) - np.sqrt(3*1)) / (np.sqrt(20*6) - np.sqrt(3*1))
    scaler = (max(xlength,ylength) - max(3,1)) / (max(20,6) - max(3,1))
    print(scaler)
    # scaler = 2*(xlength*ylength - 3*1) / (20*6 - 3*1)
    # plt.xlim(xmin - 2, xmax + 2)
    # plt.ylim(ymin - 2, ymax + 2)
    #axs.margins(0.2)
    
    def drawbeam(beam: Beam):
        plt.plot([beam.x1, beam.x2], [beam.y1, beam.y2], color='black', linewidth=2)
        amin = 0.05
        amax = 0.4
        a = amin + scaler * (amax - amin)
        rmin = 0.15*0.3
        rmax = 0.6*0.3
        r = rmin + scaler * (rmax - rmin)
        for hinge, is_begin in beam.hinges:
            h = Circle([hinge.x - (1 - 2*int(is_begin))* a*np.cos(np.radians(beam.angle)), hinge.y - (1 - 2*int(is_begin))* a*np.sin(np.radians(beam.angle))], radius=r, facecolor='white',edgecolor='black', zorder=2)
            axs.add_patch(h)
        midx = (beam.x1+beam.x2)/2
        midy = (beam.y1+beam.y2)/2

        lmin = 0.5
        lmax = 0.5
        l = lmin + scaler * (lmax - lmin)
        if beam.label is not None:
            x = midx
            if beam.labelx == 'left':
                x -= l
            if beam.labelx == 'right':
                x += l

            y = midy
            if beam.labely == 'top':
                y += l
            if beam.labely == 'bottom':
                y -= l
            axs.annotate(text=beam.label, xy=(x,y), ha='center', va='center')
            
        if beam.anglelabel:
            dx1 = (1 - 2*int(beam.labelflip)) * (beam.x2 - beam.x1) / 4
            dy1 = (1 - 2*int(beam.labelflip)) * (beam.y2 - beam.y1) / 4
            dx = int(abs(beam.x2 - beam.x1)*100) 
            dy = int(abs(beam.y2 - beam.y1)*100) 
            gcd = np.gcd(dx, dy)
            dx = int(dx/gcd)
            dy = int(dy/gcd)
            plt.plot([midx-dx1/2,midx+dx1/2,midx+dx1/2],[midy-dy1/2,midy-dy1/2,midy+dy1/2],linewidth=1,color="black")
            umin = 0.2
            umax = 0.6
            u = umin + scaler*(umax-umin)
            axs.annotate(text=str(dy), xy=(midx + 0.5*dx1 + u*dx1/abs(dx1), midy), ha='center',va='center')
            axs.annotate(text=str(dx), xy=(midx, midy - 0.5*dy1 - u*dy1/abs(dy1)), ha='center',va='center')

    for b in structure._beams:
        drawbeam(b)

    def drawpoint(point: Point):
        lmin = 0.2
        lmax = 0.5
        l = lmin + scaler * (lmax - lmin)
        x = point.x
        if point.labelx == 'left':
            x -= l
        if point.labelx == 'right':
            x += l

        y = point.y
        if point.labely == 'top':
            y += l
        if point.labely == 'bottom':
            y -= l 

        axs.annotate(text=point.label, xy=(x,y), ha='center', va='center')

    for p in structure._points:
        drawpoint(p)

    def drawhinge(hinge):
        rmin = 0.15
        rmax = 0.6
        r = rmin + scaler*(rmax-rmin)
        h = Circle([hinge.x,hinge.y], radius=r*0.3, facecolor='white',edgecolor='black', zorder=2)
        axs.add_patch(h)

    for h in structure._hinges:
        drawhinge(h)

    def drawmoment(moment: Moment):
        angle = moment.angle
        rmin = 0.5
        rmax = 2.5
        radius = rmin + scaler * (rmax - rmin)
        m = Arc((moment.point.x, moment.point.y), width=radius, height = radius, angle=angle, theta1=0, theta2=150, color=moment.color, alpha=1, linewidth=2)
        axs.add_patch(m)
        
        arrowhead = FancyArrow(
                    moment.point.x + radius * 0.5 * np.cos(np.radians(1*(angle + 10+(1-int(moment.clock_wise))*150))),
                    moment.point.y + radius * 0.5 * np.sin(np.radians(1*(angle + 10+(1-int(moment.clock_wise))*150))),
                    (2*int(moment.clock_wise)-1)* 0.1 * np.cos(np.radians(angle + 10 + (1-int(moment.clock_wise))*150 - 90)), #moet groter worden met de radius
                    (2*int(moment.clock_wise)-1)* 0.1 * np.sin(np.radians(angle + 10 + (1-int(moment.clock_wise))*150 - 90)),
                    width=0.02*radius/rmin,
                    head_width=0.1*radius/rmin,
                    head_length=0.15*radius/rmin,
                    color=moment.color,
                    alpha=1.0,
                    length_includes_head=True
                )
        axs.add_patch(arrowhead)
        umin = 0.2
        umax = 0.6
        u = umin + scaler * (umax-umin)
        x = moment.point.x 
        if moment.labelx == 'left':
            x -= 0.5*radius + u 
        if moment.labelx == 'right':
            x += 0.5*radius + u

        y = moment.point.y
        if moment.labely == 'top':
            y += 0.5*radius + u
        if moment.labely == 'bottom':
            y -= 0.5*radius + u 
        
        if moment.alt_label is not None:
            axs.annotate(text=moment.alt_label, xy=(x,y), ha='center', va='center')
        elif moment.value is not None:
            axs.annotate(text=str(moment.value) + ' ' + moment.unit, xy=(x,y), ha='center', va='center')


    for h in structure._moments:
        drawmoment(h)

    def drawpointload(pointload: PointLoad):
        lmin = 0.8
        lmax = 2.5
        length = lmin + scaler*(lmax-lmin) # afhankelijk van andere pointloads en totale grootte van de structure
        if pointload.anglelabel:
            length = 2 * length # if label is wanted, extend the arrow so there is room for the label
        tip = (pointload.x, pointload.y)
        ddx = pointload.dx * length / np.sqrt(pointload.dx**2 + pointload.dy**2)
        ddy = pointload.dy * length / np.sqrt(pointload.dx**2 + pointload.dy**2)
        start = (pointload.x + ddx, pointload.y + ddy) # depends on angle and wanted length

        axs.annotate(text='', xy=tip, xytext=start, arrowprops=dict(arrowstyle='simple',color=pointload.color))

        umin = 0.1
        umax = 0.1
        u = umin + scaler * (umax - umin)
        x = start[0]
        if pointload.labelx == 'left':
            x -= 2*u * length
        if pointload.labelx == 'right':
            x += 2*u * length

        y = start[1]
        if pointload.labely == 'top':
            y += u * length
        if pointload.labely == 'bottom':
            y -= u * length 
        if pointload.value is not None:
            axs.annotate(text=str(pointload.value) + ' ' + pointload.unit, xy=(x,y), ha='center', va='center')
        else:
            axs.annotate(text=pointload.alt_label, xy=(x,y), ha='center', va='center')
    
        if pointload.anglelabel: 
            dx1 = (1 - 2*int(pointload.labelflip)) * ddx/3
            dy1 = (1 - 2*int(pointload.labelflip)) * ddy/3
            midx = (tip[0] + start[0])/2
            midy = (tip[1] + start[1])/2
            dx = int(abs(pointload.dx)*100)
            dy = int(abs(pointload.dy)*100)
            gcd = np.gcd(dx, dy)
            dx = int(dx/gcd)
            dy = int(dy/gcd)
            umin = 0.2
            umax = 0.6
            u = umin + scaler*(umax-umin)
            plt.plot([midx-dx1/2,midx+dx1/2,midx+dx1/2],[midy-dy1/2,midy-dy1/2,midy+dy1/2],linewidth=1,color="black")
            axs.annotate(text=str(dy), xy=(midx + 0.5*dx1 + u*dx1/abs(dx1), midy), ha='center',va='center')
            axs.annotate(text=str(dx), xy=(midx, midy - 0.5*dy1 - u*dy1/abs(dy1)), ha='center',va='center')

    
    for p in structure._pointloads:
        print(p)
        drawpointload(p)

    def drawdistributedload(dload: DistributedLoad):
        lmin = 0.6
        lmax = 2.5
        length_mid = lmin + scaler * (lmax - lmin)
        beam_length = dload.begin.distance_to(dload.end)
        beam_angle = dload.end.angle(dload.begin)
        n_arrow = dload.n_arrow
        dist = beam_length/(n_arrow - 1)
        v_mid = (dload.begin_value + dload.end_value)/2
        length = length_mid / v_mid * np.linspace(dload.begin_value, dload.end_value, n_arrow)
        # plot line
        plt.plot([dload.begin.x + 0.95 * length[0] * np.cos(np.radians(dload.angle)), dload.end.x + 0.95 * length[-1] * np.cos(np.radians(dload.angle))], 
                 [dload.begin.y + 0.95 * length[0] * np.sin(np.radians(dload.angle)), dload.end.y + 0.95 * length[-1] * np.sin(np.radians(dload.angle))], 
                 color=dload.color, linewidth=2)
        # plot arrows
        for i in range(n_arrow):
            tip = (dload.begin.x + i * dist * np.cos(np.radians(beam_angle)), 
                   dload.begin.y + i * dist * np.sin(np.radians(beam_angle)))
            start = (dload.begin.x + i * dist * np.cos(np.radians(beam_angle)) + length[i] * np.cos(np.radians(dload.angle)), 
                     dload.begin.y + i * dist * np.sin(np.radians(beam_angle)) + length[i] * np.sin(np.radians(dload.angle)))
            if np.sqrt((tip[0]-start[0])**2+(tip[1]-start[1])**2) > 0.1:
                axs.annotate(text='', xy=tip, xytext=start, arrowprops=dict(arrowstyle='simple',color=dload.color))

        # display text at begin point
        x = dload.begin.x + length[0] * np.cos(np.radians(dload.angle))       
        if dload.labelx == 'left':
            x -= 0.4 * length_mid
        if dload.labelx == 'right':
            x += 0.4 * length_mid 

        y = dload.begin.y +  length[0] * np.sin(np.radians(dload.angle)) 
        if dload.labely == 'top':
            y += 0.2 * length_mid 
        if dload.labely == 'bottom':
            y -= 0.2 * length_mid 
        
        if dload.alt_label_begin is None:
            axs.annotate(text=str(dload.begin_value) + ' ' + dload.unit, xy = (x,y), ha='center', va='center')
        else:
            axs.annotate(text=dload.alt_label_begin, xy = (x,y), ha='center', va='center')

        # display text at end point
        x = dload.end.x + length[-1] * np.cos(np.radians(dload.angle))       
        if dload.labelx_end == 'left':
            x -= 0.4 * length_mid
        if dload.labelx_end == 'right':
            x += 0.4 * length_mid 

        y = dload.end.y +  length[-1] * np.sin(np.radians(dload.angle)) 
        if dload.labely_end == 'top':
            y += 0.2 * length_mid 
        if dload.labely_end == 'bottom':
            y -= 0.2 * length_mid 
        
        if dload.alt_label_end is None:
            axs.annotate(text=str(dload.end_value) + ' ' + dload.unit, xy = (x,y), ha='center', va='center')
        else:
            axs.annotate(text=dload.alt_label_end, xy = (x,y), ha='center', va='center')

    for d in structure._distributedloads:
        drawdistributedload(d)

    def drawfixedsupport(support: Support):
        angle = support.angle
        #angle = 30
        amin = 0.2
        amax = 0.8
        a = amin + scaler * (amax-amin)
        thickness = 15
        size = 100
        size2 = 200
        pattern = np.zeros((size, size2))
        for j in range(size):
            for i in range(size2):
                if ((i + j) // thickness) % 2 == 0:
                    pattern[j, i] = 1  
        pattern_rotated = rotate(pattern, -angle, reshape=True, order=0, mode='constant', cval=0)
        pattern = np.flipud(pattern)
        # Plot the pattern using imshow()
        original_width = 2*a
        original_height = a
        new_width = abs(original_width * np.cos(np.radians(angle))) + abs(original_height * np.sin(np.radians(angle)))
        new_height = abs(original_width * np.sin(np.radians(angle))) + abs(original_height * np.cos(np.radians(angle)))

        axs.imshow(pattern_rotated, 
                   cmap="gray_r", 
                   extent=[support.x - 0.5*(new_width - a*np.sin(np.radians(angle))), 
                    support.x + 0.5*(new_width + a*np.sin(np.radians(angle))), 
                    support.y - 0.5*(new_height + a*np.cos(np.radians(angle))), 
                    support.y + 0.5*(new_height - a*np.cos(np.radians(angle)))], 
                   aspect='equal', 
                   origin='lower')
        #axs.set_xlim(xmin - 2, xmax + 2)
        #axs.set_ylim(ymin - 2, ymax + 2)
        #axs.margins(0.2)
        # Draw a line next to the rectangle
        axs.plot([support.x - a*np.cos(np.radians(angle)), 
                support.x + a*np.cos(np.radians(angle))], 
                [support.y - a*np.sin(np.radians(angle)), 
                support.y + a*np.sin(np.radians(angle))], 
                color='black', linewidth=2)
        
    for f in structure._fixedsupports:
        drawfixedsupport(f)

    def drawrollersupport(support: Support):
        basetriangle = np.array([[0,0], [-0.67, -1], [0.67, -1], [0,0]])
        baseline1 = np.array([[-1, -1], [1, -1]])
        baseline2 = np.array([[-1, -1.2], [1, -1.2]])
        # scaling
        smin = 0.2
        smax = 0.8
        scaling = smin + scaler * (smax - smin)
        scaledtriangle = basetriangle @ np.array([[scaling, 0], [0, scaling]]).T
        scaledline1 = (baseline1 - (0,-1)) @ np.array([[scaling, 0], [0, scaling]]).T + (0,-1*scaling)
        scaledline2 = (baseline2 - (0,-1.2)) @ np.array([[scaling, 0], [0, scaling]]).T + (0,-1.2*scaling)
        # rotating
        angle = np.radians(support.angle) # in radialen
        rotatedtriangle = scaledtriangle @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotatedline1 = scaledline1 @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotatedline2 = scaledline2 @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # put support at the right point
        shiftedtriangle = rotatedtriangle + (support.x,support.y) # move to wanted point
        shiftedline1 = rotatedline1 + (support.x,support.y)
        shiftedline2 = rotatedline2 + (support.x,support.y)
        # make patches and add them
        triangle = Polygon(shiftedtriangle, facecolor = '#00CC99',edgecolor='black')
        line1 = Polygon(shiftedline1, fill=False, edgecolor='black', linewidth=1)
        line2 = Polygon(shiftedline2, fill=False, edgecolor='black', linewidth=1)
        axs.add_patch(triangle)
        axs.add_patch(line1)
        axs.add_patch(line2)
    
    for r in structure._rollersupports:
        drawrollersupport(r) 

    def drawpinnedsupport(support: Support):
        basetriangle = np.array([[0,0], [-0.67, -1], [0.67, -1], [0,0]])
        baseline1 = np.array([[-1, -1], [1, -1]])
        # scaling
        smin = 0.2
        smax = 0.8
        scaling = smin + scaler * (smax - smin)
        scaledtriangle = basetriangle @ np.array([[scaling, 0], [0, scaling]]).T
        scaledline1 = (baseline1 - (0,-1)) @ np.array([[scaling, 0], [0, scaling]]).T + (0,-1*scaling)
        # rotating
        angle = np.radians(support.angle) # in radialen
        rotatedtriangle = scaledtriangle @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotatedline1 = scaledline1 @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # put support at the right point
        shiftedtriangle = rotatedtriangle + (support.x,support.y) # move to wanted point
        shiftedline1 = rotatedline1 + (support.x,support.y)
        # make patches and add them
        triangle = Polygon(shiftedtriangle, facecolor = '#00CC99',edgecolor='black')
        line1 = Polygon(shiftedline1, fill=False, edgecolor='black', linewidth=1)
        axs.add_patch(triangle)
        axs.add_patch(line1)

    for p in structure._pinnedsupports:
        drawpinnedsupport(p)

    def drawrotationspring(rspring):
        theta = np.radians(np.linspace(2,360*2,1000))
        rmin = 0.025
        rmax = 0.08
        r = theta**0.7 * (rmin + scaler * (rmax-rmin) )
        x_2 = r*np.cos(theta) + rspring.x
        y_2 = r*np.sin(theta) + rspring.y
        plt.plot(x_2,y_2, color='black', linewidth=1)

    for r in structure._rotationsprings:
        drawrotationspring(r)

    def drawtranslationspring(tspring):
        ... 

    for t in structure._translationsprings:
        drawtranslationspring(t)
    
    def drawlength(length: Length):
        if length.pos == 'x':
            x1 = length.point1.x
            x2 = length.point2.x
            uymin = 0.6
            uymax = 2.2
            uy = uymin + scaler * (uymax - uymin)
            if length.xpos == 'bottom':
                y = ymin - uy
            else:
                y = ymax + uy
            text = str(round(abs(x2 - x1), 1)) + ' m' if length.altlabel is None else length.altlabel
            axs.annotate(text='', xy=(x1,y), xytext=(x2,y), arrowprops=dict(arrowstyle='<->',shrinkA=0,shrinkB=0))
            umin = 0.2
            umax = 1
            u = umin +scaler*(umax-umin)
            axs.annotate(text=text,xy=((x2+x1)/2,y+u),ha='center',va='top')
        else:
            y1 = length.point1.y
            y2 = length.point2.y
            uxmin = 0.8
            uxmax = 2
            ux = uxmin + scaler * (uxmax - uxmin)
            if length.ypos == 'left':
                x = xmin - ux
            else:
                x = xmax + ux
            text = str(round(abs(y2 - y1), 1)) + ' m' if length.altlabel is None else length.altlabel
            axs.annotate(text='', xy=(x,y1), xytext=(x,y2), arrowprops=dict(arrowstyle='<->',shrinkA=0,shrinkB=0))
            umin = 0.3
            umax = 1
            u = umin +scaler*(umax-umin)
            axs.annotate(text=text,xy=(x+u, (y2+y1)/2),ha='center',va='center')

    for l in structure._lengths:
        drawlength(l)
    axs.use_sticky_edges = False
    axs.autoscale()
    fig = plt.gcf()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    def artist_bbox(artist):
        if not artist.get_visible() or artist in {axs.patch, fig.patch}:
            return None
        if artist.__class__.__name__ in {'Spine', 'XAxis', 'YAxis'}:
            return None
        if artist.__class__.__name__ == 'Annotation' and artist.get_text() == '':
            getter_order = ('get_tightbbox',)
        else:
            getter_order = ('get_tightbbox', 'get_window_extent')

        for getter_name in getter_order:
            try:
                bbox = getattr(artist, getter_name)(renderer)
            except Exception:
                continue
            if bbox is not None and bbox.width > 0 and bbox.height > 0:
                # Empty-text annotations can return a default tiny box at the origin.
                if bbox.x0 == 0 and bbox.y0 == 0 and bbox.width <= 1 and bbox.height <= 1:
                    continue
                return bbox
        return None

    artist_bboxes = [bbox for artist in axs.get_children() if (bbox := artist_bbox(artist)) is not None]

    if artist_bboxes:
        content_bbox = Bbox.union(artist_bboxes).transformed(fig.dpi_scale_trans.inverted())
    else:
        content_bbox = fig.get_tightbbox(renderer)
        assert content_bbox is not None
    content_bbox = content_bbox.padded(0.1)
    if seed is not None:
        constructiehash = hashlib.sha256(seed.encode()).hexdigest()
        fig.savefig(constructiehash+'.svg', format='svg', bbox_inches=content_bbox)
    plt.show()
    




# TODO:
# MVN-lijnen
# translatieveer toevoegen
# 3D
# kabel, parabool toevoegen aan gewone 
# doorsnede plot

## Later:
# - Hinge in beam gaat niet altijd de goede kant op: wss verschilt het of het wel of niet het begin van de balk is
# - Label positie automatisch vinden
# radius


A = Point(0,0, 'A', labelpos=('top', 'left'))
B = Point(A.x+1.5,8, 'B', labelpos=('top', 'right'))
As = Support(A, 'pinned', angle=90)
Bs = Support(B, 'pinned', angle=90)
AB = Beam(A, B, anglelabel=True)
L = Length(A, B, ax='y', ypos='right')
F = DistributedLoad(A, B, 40, 40, angle=AB.angle-90)
st = Structure()
st.add_beam(AB)
st.add_support(As)
st.add_support(Bs)
st.add_length(L)
st.add_distributedload(F)
plot(st)
