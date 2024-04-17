# -*- coding: utf-8 -*-
"""
AlfaBeta.py

Calculates the runout distance of snow avalanches and landslides using the Alfa
Beta method (Lied & Bakkehøi 1980; Bakkehøi et al. 1983; Lied & Kristensen 2003;
Norem og Sandersen 2012).

The profil function is modified from Jan Helge Aalbu's alfabeta
(https://github.com/jhaalbu/alfabeta)

Developped in March 2024 by Pierrick Nicolet, at NVE (psni@nve.no)
"""

import arcpy
import os
import sys
import numpy as np
import pandas as pd


def FindBetaFromLine(profile_lyr,beta_lyr):
    # Intersect the beta lines with the profiles and return points

    arcpy.Intersect_analysis(in_features=[[profile_lyr, 1], [beta_lyr, 2]], out_feature_class=r"memory\intersections", output_type="POINT")
    
    return r"memory\intersections"

def profil(id_profile,inputfc, terreng, outputfc=r"memory\punkter"):
    # modified from Jan Helge Aalbu's alfabeta (https://github.com/jhaalbu/alfabeta)
    
    # Create points every meter along the profile
    arcpy.GeneratePointsAlongLines_management(inputfc, outputfc, 'DISTANCE',
                                          Distance='1 Meters')
    # Extract the altitude for each point of the profile
    arcpy.ddd.AddSurfaceInformation(outputfc, terreng, 'Z', 'BILINEAR')
    
    # Convert the profile from a feature class to a list
    hasNone = False
    with arcpy.da.SearchCursor(outputfc, ["SHAPE", 'Z']) as cursor:
        x_list = []
        y_list = []
        z_list = []
        for row in cursor:
            # Only include the points with altitude data
            # Designed to avoid errors if the profile is too long on the run-out side
            # Could be a problem if the source area has no data
            if not row[1] is None:
                x, y = row[0]
                x_list.append(x)
                y_list.append(y)
                z_list.append(row[1])
            else:
                hasNone = True
        
    # Convert the the profile from a list to a pandas dataframe
    df = pd.DataFrame(list(zip(x_list, y_list, z_list)), columns =['X', 'Y', 'Z'])
    
    if len(df) > 0:
        if hasNone:
            arcpy.AddWarning('Profile {} extends outside of the DTM layer. Only the points with altitude value are considered further'.format(id_profile))
        
        #Calculates the lenght along the profile (1 meter between points)
        df = df.assign(M=list(range(0,len(df))))
        
        # Calculate the angle to the fisrt point in degrees
        df.loc[1:,'H'] = np.rad2deg(np.arctan((df.at[0,'Z'] - df.loc[1:,'Z']) / (df.loc[1:,'M'] - df.at[0,'M'])))
    
        if df.at[len(df)-1,'Z'] > df.at[0,'Z']:
            arcpy.AddWarning('Check if profile {} is oriented correctly. It must start form the release area'.format(id_profile))
    else:
        arcpy.AddWarning('Profile {} is completely outside the DTM layer and will therfore not be computed'.format(id_profile))
    
    return df

def calculate_profile(id_profile,df_profile,betas):
    
    # Create a search cursor in the layer with beta points
    b_rows = arcpy.da.SearchCursor(betas,["FID_{}".format(os.path.basename(profile_lyr)),'SHAPE@XY'])
    b_xy = []
    
    #Find the beta point corresponding to the profile assuming only one intersection
    for b_row in b_rows:
        if b_row[0] == id_profile:
            b_xy = b_row[1]
    
    if len(b_xy) > 0:
        arcpy.AddMessage('Computing profile {}'.format(id_profile))
        
        # Calculates the distance of each point of the profile to the beta point
        df_profile['Dist2B'] = np.sqrt(((df_profile.X - b_xy[0])**2)+((df_profile.Y - b_xy[1])**2))
    
        # Find the 2 closest points to beta
        pts = df_profile['Dist2B'].nsmallest(2)
        
        # Calculates the contribution of each points (probably overdriven, the closest point would be good enough)
        f1 = 1 - (pts.to_numpy()[0]/(pts.to_numpy()[0] + pts.to_numpy()[1]))
        f2 = 1 - (pts.to_numpy()[1]/(pts.to_numpy()[0] + pts.to_numpy()[1]))
        
        # Calculates the height, distance and angle to the beta point
        b_z = (f1 * df_profile.at[(pts.index.tolist()[0]),'Z']) + (f2 * df_profile.at[(pts.index.tolist()[1]),'Z'])
        b_m = (f1 * df_profile.at[(pts.index.tolist()[0]),'M']) + (f2 * df_profile.at[(pts.index.tolist()[1]),'M'])
        b_h = np.rad2deg(np.arctan((df_profile.at[0,'Z'] - b_z) / (b_m - df_profile.at[0,'M'])))
        
        # Calculate the alpha and sigma angles
        alpha, alpha_s1, alpha_s2 = vinkler(b_h, faretype)
        
        # Calculates the height difference between each point of the profile and the alfa/sigmas lines
        df_profile['dz_alpha'] = (df_profile.at[0,'Z'] - (df_profile['M'] * np.tan(np.deg2rad(alpha)))) - df_profile['Z']
        df_profile['dz_alpha_s1'] = (df_profile.at[0,'Z'] - (df_profile['M'] * np.tan(np.deg2rad(alpha_s1)))) - df_profile['Z']
        df_profile['dz_alpha_s2'] = (df_profile.at[0,'Z'] - (df_profile['M'] * np.tan(np.deg2rad(alpha_s2)))) - df_profile['Z']
    
        # Finds the intersections iteratively below the beta point
        i_beta = pts.index.tolist()[0]
        i = i_beta

        # Finds the first point after beta that is below the alfa line
        while (df_profile.at[i,'dz_alpha'] > 0) and (i < len(df_profile)-1):
            i += 1
        if i < len(df_profile)-1:
            i_alpha = i
        else:
            i_alpha = np.nan
        
        # Finds the first point after beta that is below the sigma 1 line
        while (df_profile.at[i,'dz_alpha_s1'] > 0) and i < len(df_profile)-1:
            i += 1
        if i < len(df_profile)-1:
            i_alpha_s1 = i
        else:
            i_alpha_s1 = np.nan
        
        # Finds the first point after beta that is below the sigma 2 line
        while (df_profile.at[i,'dz_alpha_s2'] > 0) and i < len(df_profile)-1:
            i += 1
        if i < len(df_profile)-1:
            i_alpha_s2 = i
        else:
            i_alpha_s2 = np.nan
    
        # Open an insert cursor in the output table
        cursor = arcpy.da.InsertCursor(outPts, ['SHAPE@XY'] + field_names)
        
        #arcpy.AddMessage('Points locatd at: beta={}, alpha={}, 1 sigma={}, 2sigma={}'.format(i_beta,i_alpha,i_alpha_s1,i_alpha_s2))
        
        # Adds the points to the table
        for name, i_pt, h in zip(['Beta','Alpha','Sigma_1','Sigma_2'],[i_beta,i_alpha,i_alpha_s1,i_alpha_s2],[b_h,alpha,alpha_s1,alpha_s2]):
            if np.isfinite(i_pt):
                cursor.insertRow([(df_profile.at[i_pt,'X'],df_profile.at[i_pt,'Y'])]
                + [id_profile,name,df_profile.at[i_pt,'X'],
                   df_profile.at[i_pt,'Y'],
                   df_profile.at[i_pt,'Z'],
                   df_profile.at[i_pt,'M'],
                   h])
        
        # Saves a figure (optional)
        if len(outFolder) > 0:
            # plot profile
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(df_profile['M'],df_profile['Z'],'k')
            for i_pt, color, h in zip([i_beta,i_alpha,i_alpha_s1,i_alpha_s2],[':k','r','darkorange','yellow'],[b_h,alpha,alpha_s1,alpha_s2]):
                if np.isfinite(i_pt):
                    ax.plot([df_profile.at[0,'M'],df_profile.at[i_pt,'M']],[df_profile.at[0,'Z'],df_profile.at[i_pt,'Z']],color)
                else:
                    ax.plot([df_profile.at[0,'M'],df_profile.at[len(df)-1,'M']],[df_profile.at[0,'Z'],df_profile.at[0,'Z'] - (df_profile.at[len(df_profile)-1,'M'] * np.tan(np.deg2rad(h)))],color)
            ax.axis('equal')
            ax.axis('tight')
            ax.grid(linestyle=':')
            fig.savefig(os.path.join(outFolder,'profile_{}.png'.format(id_profile)))
            plt.close()
        
        # Delete cursor
        del cursor

def vinkler(b_h, faretype):
    
    if faretype == 'Snøskred':
        x1 = 0.96
        x2 = -1.4
        s = 2.3
    elif faretype == 'Steinsprang':
        x1 = 0.77
        x2 = 3.9
        s = 2.16
    elif faretype == 'Jordskred':
        x1 = 0.96
        x2 = -4.0
        s = 1.5
    elif faretype == 'Skredvind':
        x1 = 0.79
        x2 = 1.7
        s = 2.3 #Kanskje ikke den riktige verdien
        
    alpha = (x1 * b_h) + x2
    alpha_s1 = alpha - s
    alpha_s2 = alpha - (2 * s)
    
    return alpha,alpha_s1,alpha_s2

if __name__=="__main__":
    profile_lyr = arcpy.GetParameterAsText(0)
    beta_lyr = arcpy.GetParameterAsText(1)
    dtm = arcpy.GetParameterAsText(2)
    faretype = arcpy.GetParameterAsText(3)
    outPts = arcpy.GetParameterAsText(4)
    outFolder = arcpy.GetParameterAsText(5)
    
    savefigs = True
    
    # TODO: Check the projections
    
    try:
        profile_desc = arcpy.Describe(profile_lyr)
        try: #works if the input is a layer
            Set = profile_desc.FIDSet
        except: #makes a layer first if the input is a file
            if arcpy.GetInstallInfo()['ProductName'] == 'Desktop':
                profile_lyr = arcpy.mapping.Layer(profile_lyr)
            else:
                profile_lyr = arcpy.mp.Layer(profile_lyr)
            profile_desc = arcpy.Describe(profile_lyr)
            Set = profile_desc.FIDSet

        if profile_desc.shapeType != "Polyline":
            arcpy.AddError('The profile layer must be a polyline')
        else:
            # Check the data type for beta
            try:
                beta_desc = arcpy.Describe(beta_lyr)
                try: #works if the input is a layer
                    BSet = beta_desc.FIDSet
                except: #makes a layer first if the input is a file
                    if arcpy.GetInstallInfo()['ProductName'] == 'Desktop':
                        beta_lyr = arcpy.mapping.Layer(beta_lyr)
                    else:
                        beta_lyr = arcpy.mp.Layer(beta_lyr)
                    beta_desc = arcpy.Describe(beta_lyr)
                    BSet = beta_desc.FIDSet
                if beta_desc.shapeType != "Polyline":
                    arcpy.AddError('The beta layer must be a polyline')
            except:
                pass
            
            OID_name = profile_desc.OIDFieldName
            rows = arcpy.da.SearchCursor(profile_lyr,[OID_name,'SHAPE@'])
            
            betas = FindBetaFromLine(profile_lyr,beta_lyr)
            
            if int(arcpy.GetCount_management(betas)[0]) == 0:
                arcpy.AddError("No intersection found between the profiles and the beta line. Check the input data.")
            else:
                # Creates the output feature class or shapefile
                arcpy.CreateFeatureclass_management(os.path.dirname(outPts),os.path.basename(outPts), "POINT",spatial_reference=profile_desc.spatialReference)
            
                # Define field names and types
                field_names = ['Profile','Name','X','Y','Z','M','H']
                field_types = ['TEXT','TEXT','DOUBLE','DOUBLE','DOUBLE','DOUBLE','DOUBLE']
                
                # Add fields to the output table
                for field_name, field_type in zip(field_names, field_types):
                    arcpy.AddField_management(outPts, field_name, field_type)
                
                # Check if some profiles are selected
                if len(Set) > 0:
                    for row in rows:
                        feat = row[0]
                        if str(feat) in Set:
                            # calculate profile
                            df = profil(row[0],row[1], dtm)
                            if len(df) > 0:
                                calculate_profile(row[0],df,betas)
                        else:
                            pass
                else: #no selected profile --> takes all
                    for row in rows:
                        # calculate profile
                        df = profil(row[0],row[1], dtm)
                        
                        if len(df) > 0:
                            calculate_profile(row[0],df,betas)
        
        arcpy.management.Delete(r"memory\intersections")
        
        try:
            mxd = arcpy.mp.ArcGISProject("CURRENT")
            dataFrame = mxd.listMaps("*")[0]
            lyr = dataFrame.addDataFromPath(outPts)
            
            try:
                # Change symbology
                sym = lyr.symbology
                sym.updateRenderer('UniqueValueRenderer')
                sym.renderer.fields = ['Name']
                
                # applying symbology to each defined value
                for grp in sym.renderer.groups:
                    for itm in grp.items:
                        myVal = itm.values[0][0]
                        
                        itm.symbol.applySymbolFromGallery("Circle 3",1) #It seems that the second "Circle 3" is the right one
                        itm.symbol.outlineColor = {"RGB": [0, 0, 0, 100]}
                        itm.symbol.size = 5
            
                        if myVal == "Beta":
                            itm.symbol.color = {"RGB": [56, 168, 0, 100]}
                            itm.label = str(myVal)
            
                        elif myVal == "Alpha":
                            itm.symbol.color = {"RGB": [255, 0, 0, 100]}
                            itm.label = str(myVal)
    
                        elif myVal == "Sigma_1":
                            itm.symbol.color = {"RGB": [255, 170, 0, 100]}
                            itm.label = "1 sigma"
                            
                        elif myVal == "Sigma_2":
                            itm.symbol.color = {"RGB": [255, 255, 0, 100]}
                            itm.label = "2 sigma"
            except:
                arcpy.AddMessage('Output file has been added, but the symbology could not be updated')
            
            # save symbology back onto layer
            lyr.symbology = sym
            
        except:
            arcpy.AddMessage('Output file has been saved, but could not be added to the project. Try importing it manualy from {}'.format(outPts))
    
    except:
        arcpy.AddError("Unexpected error")
        e = sys.exc_info()[1]
        arcpy.AddError(e.args[0])
        arcpy.Delete_management("in_memory")
        sys.exit(0)
