# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:52:17 2023

@author: TranK
"""
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

def random_points1D_withmindst(low_bound, up_bound, mindst, npoints):
    # snippet code taken from https://stackoverflow.com/questions/53565205/how-to-generate-random-numbers-with-each-random-number-having-a-difference-of-at
    space = up_bound - low_bound - (npoints-1)*mindst
    assert space >= 0 
    
    coef = np.random.rand(npoints)
    a = space * np.sort(coef, axis=-1)
    return np.ceil(low_bound + a + mindst * np.arange(npoints)).astype(int)

###Code below inspired heavily from : https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib 

def bernstein(n, i, t):
    return binom(n,i)*(1.-t)**(n-i)*t**i 

def bezier_explicit(controlled_points, num=200):
    N = len(controlled_points)
    t = np.linspace(0, 1, num=num)
    bezier_curve = np.zeros((num, 2))
    for i in range(N):
        bezier_curve += np.outer(bernstein(N - 1, i, t), controlled_points[i])
    return bezier_curve

def sort_counterClockWise(a):
    #sort an array of 2D points by counter clock wise order
    #center point is the mean of all points
    dst_from_mean = a-np.mean(a,axis=0)
    
    #find the angle between each point and the center point
    #chosing right quadrant with arctan2(x1, x2) for counter-clockwise order
    angle = np.arctan2(dst_from_mean[:,0], dst_from_mean[:,1])
    
    #index that would sort angle array from low to high value 
    #in case of tie, it follows the index of points in initial array
    index_sort = np.argsort(angle)
    
    return a[index_sort,:]

def interpolate_bezier(a, radius_coef=0.2, angle_combine_factor = 0.5):
        
    #sort counterclockwise 
    a = sort_counterClockWise(a)
    
    #add first point to last index to have a cycle
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    
    #calculate angles linking adjacent points
    adjacent_vect = np.diff(a, axis=0)
    angle = np.arctan2(adjacent_vect[:,1],adjacent_vect[:,0]) #angle between -pi and pi
    translate_angle = lambda angle : (angle>=0)*angle + (angle<0)*(angle+2*np.pi)
    angle = translate_angle(angle) #angle between [0, 2*pi[
    
    #caculate the angle as the mean of adjacent angles
    angle_before = angle
    angle_after = np.roll(angle,1)
    angle = angle_combine_factor*angle_before + (1-angle_combine_factor)*angle_after + (np.abs(angle_after-angle_before) > np.pi )*np.pi
    
    #add first point to last index to have a cycle
    angle = np.append(angle, [angle[0]])
    
    a = np.append(a, np.atleast_2d(angle).T, axis=1)
    
    #from the angles defined, calculate radius and get bezier curves cubic
    spline_segments = []
    for i in range(len(a)-1):
        start_p = a[i,:2]
        end_p = a[i+1,:2]
        
        angle_before = a[i,2]
        angle_after = a[i+1,2]
        
        #calculate radius of intermediate points
        dst = np.sqrt(np.sum((end_p - start_p)**2))
        radius = radius_coef*dst
        
        #calculate intermediate points with radius and angle known
        intermed_p_1 = start_p + np.array([radius*np.cos(angle_before),
                                    radius*np.sin(angle_before)])
        intermed_p_2 = end_p + np.array([radius*np.cos(angle_after+np.pi),
                                    radius*np.sin(angle_after+np.pi)])
        
        #get bezier curves of degree 3 with all 4 controlled points
        controlled_points = np.array([start_p, intermed_p_1, intermed_p_2, end_p])
        segment_curve = bezier_explicit(controlled_points, num=200)
        
        spline_segments.append(segment_curve)
        
    curve = np.concatenate([segment_curve for segment_curve in spline_segments])
    
    x,y = curve.T
    return x,y, a

def generate_randompoints2D_withmindistance(npoints=10, scaling_factor=1, nitermax=200):
    niter=0
    dst_min = 0.7/npoints
    while niter<nitermax:
        a = np.random.rand(npoints,2)
        dst_array = np.sqrt(np.sum(np.diff(sort_counterClockWise(a), axis=0)**2, axis=1))
        if np.all(dst_array >= dst_min):
            return a*scaling_factor
        niter+=1
    return a*scaling_factor

if 0 : 
    a = generate_randompoints2D_withmindistance(20, 9, 200)
    x, y, a = interpolate_bezier(a, radius_coef=0.3)
    
    # plt.plot(a[:, 0], a[:, 1])
    plt.plot(x, y)
    plt.show()
