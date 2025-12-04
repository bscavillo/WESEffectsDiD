import pandas as pd
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
main_data = os.path.join('daten', 'SelfEmployedTechnicalSplitByGender.csv')
control_data  = os.path.join('daten', 'labourParticipationAndUnemploymentRateSplitByGender.csv') 

main_output = os.path.join('daten', 'cleanedSelfEmployed.csv')
control_output = os.path.join('daten', 'cleanedLabourStats.csv')

def clean_number(x):
    """
    Entfernt Tausender-Kommas aus Strings, damit Python sie als Zahl lesen kann.
    Beispiel: "2,305.1" -> 2305.1
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(',', '')
    return pd.to_numeric(x, errors='coerce')

# ==============================================================================
# 1. MAIN DATA (Self-Employed)
# ==============================================================================
def process_main_data(filepath):
    
    df = pd.read_csv(
        filepath, 
        skiprows=11, 
        header=None,
        names=['Province', 'Class_of_worker', 'NAICS', 'Year', 'Men+', 'Women+']
    )

    df['Province'] = df['Province'].ffill()
    df['Class_of_worker'] = df['Class_of_worker'].ffill()
    df['NAICS'] = df['NAICS'].ffill()

    df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()].copy()
    df['Year'] = df['Year'].astype(int)

    df_long = df.melt(
        id_vars=['Province', 'Year'], 
        value_vars=['Men+', 'Women+'], 
        var_name='Sex', 
        value_name='Self_Employed_Count_1000s'
    )

    df_long['Self_Employed_Count_1000s'] = df_long['Self_Employed_Count_1000s'].apply(clean_number)
    
    df_long['Self_Employed'] = df_long['Self_Employed_Count_1000s'] * 1000

    return df_long[['Province', 'Year', 'Sex', 'Self_Employed']]

# ==============================================================================
# 2. CONTROL DATA (Labor Force)
# ==============================================================================
def parse_statcan_controls(filepath):

    df = pd.read_csv(
        filepath, 
        skiprows=10, 
        header=None, 
        names=['Province', 'Characteristic', 'Age', 'Year', 'Men+', 'Women+']
    )

    df['Province'] = df['Province'].ffill()
    df['Characteristic'] = df['Characteristic'].ffill()
    df['Age'] = df['Age'].ffill()

    df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()].copy()
    df['Year'] = df['Year'].astype(int)

    df_long = df.melt(
        id_vars=['Province', 'Year', 'Characteristic'], 
        value_vars=['Men+', 'Women+'], 
        var_name='Sex', 
        value_name='Value'
    )
    
    df_long['Value'] = df_long['Value'].apply(clean_number)

    df_pivot = df_long.pivot_table(
        index=['Province', 'Year', 'Sex'], 
        columns='Characteristic', 
        values='Value',
        aggfunc='first'
    ).reset_index()

    lf_col = [c for c in df_pivot.columns if "Labour force" in str(c)]
    ur_col = [c for c in df_pivot.columns if "Unemployment rate" in str(c)]
    
    if lf_col:
        df_pivot.rename(columns={lf_col[0]: 'Control_LaborForce'}, inplace=True)
        df_pivot['Control_LaborForce'] = df_pivot['Control_LaborForce'] * 1000
    
    if ur_col:
        df_pivot.rename(columns={ur_col[0]: 'Control_UnemploymentRate'}, inplace=True)

    return df_pivot

df_main = process_main_data(main_data)
df_controls = parse_statcan_controls(control_data)

df_main.to_csv(main_output, index=False)
df_controls.to_csv(control_output, index=False)



  
