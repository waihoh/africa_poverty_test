from typing import Any, Mapping, Optional, Tuple, Union

import ee
import pandas as pd
import time
from tqdm.auto import tqdm

Numeric = Union[int, float]


def df_to_fc(df: pd.DataFrame, lat_colname: str = 'lat', lon_colname: str = 'lon') -> ee.FeatureCollection:
    '''
    :param df: dataframe that includes at least 2 cols for latitude and longitude coordinates
    :param lat_colname: str, name of latitude column
    :param lon_colname: str, name of longitude column
    :return: ee.FeatureCollection, contains one feature per row in the dataframe
    '''

    # Convert values to Python native types. See https://stackoverflow.com/a/47424340
    df = df.astype('object')

    ee_features = []
    for i in range(len(df)):
        props = df.iloc[i].to_dict()

        # oddly EE wants (lon, lat) instead of (lat, lon)
        _geometry = ee.Geometry.Point(coords=[props[lon_colname], props[lat_colname]])
        ee_feat = ee.Feature(geom=_geometry, opt_properties=props)
        ee_features.append(ee_feat)

    return ee.FeatureCollection(ee_features)


def surveyyear_to_range(survey_year: int, nl: bool = False) -> Tuple[str, str]:
    '''
    Returns the start and end dates for filtering satellite images for a survey beginning in the specified year.

    Calibrated DMSP Nighttime Lights only exists for 3 relevant date ranges, which Google Earth Engine filters by
    their start date. For more info, see https://www.ngdc.noaa.gov/eog/dmsp/download_radcal.html.

        DMSP range               | we use for these surveys
        -------------------------|-------------------------
        2010-01-11 to 2011-07-31 | 2006 to 2011
        2010-01-11 to 2010-12-09 | 2006 to 2011
        2005-11-28 to 2006-12-24 | 2003 to 2005

    :param survey_year: int, year that survey was started
    :param nl: bool, whether to use special range for night lights
    :return:
        start_date: str, represents start date for filtering satellite images
        end_date: str, represents end date for filtering satellite images
    '''

    if 2003 <= survey_year and survey_year <= 2005:
        start_date = '2003-1-1'
        end_date = '2005-12-31'
    elif 2006 <= survey_year and survey_year <= 2008:
        start_date = '2006-1-1'
        end_date = '2008-12-31'
        if nl:
            end_date = '2010-12-31'  # artificially extend end date for DMSP
    elif 2009 <= survey_year and survey_year <= 2011:
        start_date = '2009-1-1'
        end_date = '2011-12-31'
    elif 2012 <= survey_year and survey_year <= 2014:
        start_date = '2012-1-1'
        end_date = '2014-12-31'
    elif 2015 <= survey_year and survey_year <= 2017:
        start_date = '2015-1-1'
        end_date = '2017-12-31'
    else:
        raise ValueError(f'Invalid survey_year: {survey_year}. Must be between 2009 and 2017 (inclusive)')

    return start_date, end_date


def decode_qamask(img: ee.Image) -> ee.Image:
    '''
    Pixel QA Bit Flags (universal across Landsat 5/7/8)
    Bit  Attribute
    0    Fill
    1    Clear
    2    Water
    3    Cloud Shadow
    4    Snow
    5    Cloud
    :param img: ee.Image, Landsat 5/7/8 image containing 'pixel_qa' band
    :return: masks: ee.Image, contains 5 bands of masks
    '''

    qa = img.select(opt_selectors='pixel_qa')
    clear = qa.bitwiseAnd(2).neq(0)  # 0 = not clear, 1 = clear
    clear = clear.updateMask(clear).rename(['pxqa_clear'])

    water = qa.bitwiseAnd(4).neq(0)  # 0 = not water, 1 = water
    water = water.updateMask(water).rename(['pxqa_water'])

    cloud_shadow = qa.bitwiseAnd(8).neq(0)  # 0 = shadow, 1 = no shadow
    cloud_shadow = cloud_shadow.updateMask(cloud_shadow).rename(['pxqa_cloudshadow'])

    snow = qa.bitwiseAnd(16).neq(0)  # 0 = snow, 1 = no snow
    snow = snow.updateMask(snow).rename(['pxqa_snow'])

    cloud = qa.bitwiseAnd(32).eq(0)  # 0 = cloud, 1 = not cloud
    cloud = cloud.updateMask(cloud).rename(['pxqa_cloud'])

    masks = ee.Image.cat([clear, water, cloud_shadow, snow, cloud])
    return masks

