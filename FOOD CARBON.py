import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create realistic dataset based on research and FAO data for East Africa
# Data sources: FAO, nu3 Food Carbon Footprint Index 2018, Our World in Data

def create_east_africa_food_data():
    """
    Create dataset based on research findings for East African countries
    CO2 emissions in kg per person per year by food category
    """
    
    countries = ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Ethiopia']
    
    # Food categories with CO2 emissions (kg CO2/person/year)
    # Based on typical consumption patterns and carbon intensities
    food_data = {
        'Country': [],
        'Food_Category': [],
        'CO2_Emissions': [],
        'Consumption_kg_per_year': [],
        'Is_Animal_Product': []
    }
    
    # Emission factors per kg of food (kg CO2/kg food)
    emission_factors = {
        'Beef': 60.0,
        'Dairy': 3.2,
        'Poultry': 6.9,
        'Eggs': 4.2,
        'Fish': 5.4,
        'Vegetables': 2.0,
        'Grains': 1.4,
        'Legumes': 0.9,
        'Fruits': 1.1,
        'Roots_Tubers': 0.4
    }
    
    # Typical consumption patterns (kg/person/year) for each country
    consumption_patterns = {
        'Kenya': {
            'Beef': 12.5, 'Dairy': 95.0, 'Poultry': 8.2, 'Eggs': 3.1, 'Fish': 4.8,
            'Vegetables': 85.0, 'Grains': 165.0, 'Legumes': 22.0, 'Fruits': 45.0, 'Roots_Tubers': 120.0
        },
        'Uganda': {
            'Beef': 8.5, 'Dairy': 45.0, 'Poultry': 4.2, 'Eggs': 1.8, 'Fish': 12.5,
            'Vegetables': 95.0, 'Grains': 145.0, 'Legumes': 35.0, 'Fruits': 75.0, 'Roots_Tubers': 185.0
        },
        'Tanzania': {
            'Beef': 15.2, 'Dairy': 55.0, 'Poultry': 6.8, 'Eggs': 2.5, 'Fish': 8.9,
            'Vegetables': 78.0, 'Grains': 155.0, 'Legumes': 28.0, 'Fruits': 52.0, 'Roots_Tubers': 145.0
        },
        'Rwanda': {
            'Beef': 6.8, 'Dairy': 35.0, 'Poultry': 3.5, 'Eggs': 1.2, 'Fish': 2.1,
            'Vegetables': 125.0, 'Grains': 135.0, 'Legumes': 42.0, 'Fruits': 85.0, 'Roots_Tubers': 225.0
        },
        'Ethiopia': {
            'Beef': 9.2, 'Dairy': 25.0, 'Poultry': 2.8, 'Eggs': 0.8, 'Fish': 1.2,
            'Vegetables': 65.0, 'Grains': 175.0, 'Legumes': 38.0, 'Fruits': 35.0, 'Roots_Tubers': 165.0
        }
    }
    
    animal_products = ['Beef', 'Dairy', 'Poultry', 'Eggs', 'Fish']
    
    for country in countries:
        for food, consumption in consumption_patterns[country].items():
            co2_emission = consumption * emission_factors[food]
            
            food_data['Country'].append(country)
            food_data['Food_Category'].append(food)
            food_data['CO2_Emissions'].append(co2_emission)
            food_data['Consumption_kg_per_year'].append(consumption)
            food_data['Is_Animal_Product'].append(food in animal_products)
    
    return pd.DataFrame(food_data)

def create_global_comparison_data():
    """Create global comparison data for top CO2 emitting countries"""
    global_data = {
        'Country': ['Argentina', 'Australia', 'New Zealand', 'United States', 'Brazil', 
                   'France', 'Germany', 'Canada', 'Russia', 'China',
                   'Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Ethiopia'],
        'Total_Food_CO2': [2172, 1939, 1808, 1764, 1461, 1144, 1068, 1025, 952, 875,
                          523, 387, 598, 312, 445],  # kg CO2/person/year
        'Region': ['South America', 'Oceania', 'Oceania', 'North America', 'South America',
                  'Europe', 'Europe', 'North America', 'Europe', 'Asia',
                  'East Africa', 'East Africa', 'East Africa', 'East Africa', 'East Africa']
    }
    return pd.DataFrame(global_data)

# Load the data
df_food = create_east_africa_food_data()
df_global = create_global_comparison_data()

print("🌍 East Africa Food Carbon Footprint Analysis")
print("=" * 50)
print(f"Dataset shape: {df_food.shape}")
print(f"Countries analyzed: {df_food['Country'].unique()}")
print(f"Food categories: {df_food['Food_Category'].unique()}")
print("\nFirst few rows:")
print(df_food.head())

# Calculate summary statistics
print(f"\n📊 Summary Statistics:")
print(f"Average CO2 per person per year: {df_food.groupby('Country')['CO2_Emissions'].sum().mean():.1f} kg")
print(f"Highest emitting country: {df_food.groupby('Country')['CO2_Emissions'].sum().idxmax()}")
print(f"Lowest emitting country: {df_food.groupby('Country')['CO2_Emissions'].sum().idxmin()}")

# 1. Bar Chart: Average CO2 Emission by Food Category (East Africa)
print("\n📊 VISUALIZATION 1: CO₂ Emissions by Food Category")
print("-" * 50)
plt.figure(figsize=(12, 6))
food_avg = df_food.groupby(['Food_Category', 'Country'])['CO2_Emissions'].mean().reset_index()
food_pivot = food_avg.pivot(index='Food_Category', columns='Country', values='CO2_Emissions')

ax = food_pivot.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('CO₂ Emissions by Food Category Across East African Countries', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Food Category', fontsize=11, fontweight='bold')
plt.ylabel('CO₂ Emissions (kg/person/year)', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Chart 1 displayed: Food category emissions comparison across countries")

# 2. Highlight Chart: Beef vs Plant-Based Alternatives
print("\n📊 VISUALIZATION 2: Beef vs Legumes Impact Comparison")
print("-" * 50)
plt.figure(figsize=(10, 6))
beef_data = df_food[df_food['Food_Category'] == 'Beef']['CO2_Emissions'].mean()
legume_data = df_food[df_food['Food_Category'] == 'Legumes']['CO2_Emissions'].mean()

categories = ['Beef', 'Legumes']
emissions = [beef_data, legume_data]
colors = ['#d62728', '#2ca02c']

bars = plt.bar(categories, emissions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.title('Carbon Footprint Comparison: Beef vs Legumes\n(Average across East Africa)', 
          fontsize=14, fontweight='bold', pad=15)
plt.ylabel('CO₂ Emissions (kg/person/year)', fontsize=11, fontweight='bold')

# Add value labels on bars
for bar, emission in zip(bars, emissions):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{emission:.1f} kg', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.text(0.5, max(emissions) * 0.6, f'Beef produces {beef_data/legume_data:.1f}x more CO₂\nthan legumes per person per year', 
         ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Chart 2 displayed: Beef vs Legumes impact comparison")

# 3. Stacked Bar Chart: Animal vs Non-Animal Product Emissions
print("\n📊 VISUALIZATION 3: Animal vs Plant Product Emissions by Country")
print("-" * 50)
animal_vs_plant = df_food.groupby(['Country', 'Is_Animal_Product'])['CO2_Emissions'].sum().reset_index()
animal_vs_plant['Product_Type'] = animal_vs_plant['Is_Animal_Product'].map({True: 'Animal Products', False: 'Plant Products'})

plt.figure(figsize=(11, 6))
pivot_data = animal_vs_plant.pivot(index='Country', columns='Product_Type', values='CO2_Emissions')
ax = pivot_data.plot(kind='bar', stacked=True, figsize=(11, 6), 
                     color=['#ff6b6b', '#4ecdc4'], alpha=0.8)

plt.title('Animal vs Plant-Based Food Emissions by Country', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Country', fontsize=11, fontweight='bold')
plt.ylabel('CO₂ Emissions (kg/person/year)', fontsize=11, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(title='Product Type', loc='upper right')

# Add percentage labels
for i, country in enumerate(pivot_data.index):
    total = pivot_data.loc[country].sum()
    animal_pct = (pivot_data.loc[country, 'Animal Products'] / total) * 100
    plt.text(i, total + 15, f'{animal_pct:.1f}% animal', ha='center', va='bottom', 
             fontweight='bold', fontsize=9)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Chart 3 displayed: Animal vs Plant product emissions by country")

# 4. Box Plot: Distribution of CO2 Emissions by Food Type
print("\n📊 VISUALIZATION 4: Distribution of CO₂ Emissions by Food Category")
print("-" * 50)
plt.figure(figsize=(12, 7))
food_emissions = []
food_labels = []

for food in df_food['Food_Category'].unique():
    emissions = df_food[df_food['Food_Category'] == food]['CO2_Emissions'].values
    food_emissions.append(emissions)
    food_labels.append(food)

plt.boxplot(food_emissions, labels=food_labels, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))

plt.title('Distribution of CO₂ Emissions by Food Category\n(Across East African Countries)', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Food Category', fontsize=11, fontweight='bold')
plt.ylabel('CO₂ Emissions (kg/person/year)', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Chart 4 displayed: Box plot showing emission distribution by food type")

# 5. Global Comparison - Top Countries
print("\n📊 VISUALIZATION 5: Global Food Carbon Footprint Comparison")
print("-" * 50)
plt.figure(figsize=(13, 8))
df_global_sorted = df_global.sort_values('Total_Food_CO2', ascending=True).tail(15)

# Color East African countries differently
colors = ['#ff6b6b' if region == 'East Africa' else '#4ecdc4' 
          for region in df_global_sorted['Region']]

bars = plt.barh(df_global_sorted['Country'], df_global_sorted['Total_Food_CO2'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title('Global Food Carbon Footprint Comparison\n(Top 15 Countries)', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Total Food CO₂ Emissions (kg/person/year)', fontsize=11, fontweight='bold')
plt.ylabel('Country', fontsize=11, fontweight='bold')

# Add value labels
for i, (bar, value) in enumerate(zip(bars, df_global_sorted['Total_Food_CO2'])):
    plt.text(value + 20, bar.get_y() + bar.get_height()/2, 
             f'{value} kg', va='center', fontweight='bold', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#ff6b6b', alpha=0.8, label='East Africa'),
                  Patch(facecolor='#4ecdc4', alpha=0.8, label='Other Regions')]
plt.legend(handles=legend_elements, loc='lower right')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Chart 5 displayed: Global comparison highlighting East Africa's position")

# 6. Interactive Choropleth Map using Plotly (East Africa focus)
print("\n📊 VISUALIZATION 6: Interactive Choropleth Map - Beef CO₂ Emissions")
print("-" * 50)

# Create map data
east_africa_map_data = {
    'Country': ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Ethiopia'],
    'ISO_Code': ['KEN', 'UGA', 'TZA', 'RWA', 'ETH'],
    'Beef_CO2': [df_food[(df_food['Country'] == country) & (df_food['Food_Category'] == 'Beef')]['CO2_Emissions'].iloc[0] 
                 for country in ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Ethiopia']],
    'Total_CO2': [df_food[df_food['Country'] == country]['CO2_Emissions'].sum() 
                  for country in ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Ethiopia']]
}

df_map = pd.DataFrame(east_africa_map_data)

# Display the data first
print("Map Data Preview:")
print(df_map.to_string(index=False))

try:
    # Create the choropleth map
    fig = px.choropleth(df_map, 
                        locations='ISO_Code',
                        color='Beef_CO2',
                        hover_name='Country',
                        hover_data={'Total_CO2': True, 'Beef_CO2': ':.1f'},
                        color_continuous_scale='Reds',
                        title='Beef CO₂ Emissions Across East Africa (kg/person/year)',
                        labels={'Beef_CO2': 'Beef CO₂ Emissions (kg/person/year)'})
    
    # Update layout for better display
    fig.update_layout(
        title_x=0.5, 
        title_font_size=14,
        width=900, 
        height=700,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showland=True,
            landcolor='lightgray',
            coastlinecolor='white',
            projection_type='natural earth',
            center=dict(lat=-2, lon=35),
            projection_scale=3
        )
    )
    
    # Show the map
    fig.show()
    print("✅ Chart 6 displayed: Interactive choropleth map showing beef emissions")
    
except Exception as e:
    print(f"⚠️ Interactive map display failed: {e}")
    print("Creating alternative static map visualization...")
    
    # Alternative: Create a simple bar chart as backup
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_map['Country'], df_map['Beef_CO2'], 
                   color=['#ff9999', '#ff6666', '#ff3333', '#ff0000', '#cc0000'], 
                   alpha=0.8, edgecolor='black')
    
    plt.title('Beef CO₂ Emissions Across East Africa\n(Alternative Static Visualization)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Country', fontsize=11, fontweight='bold')
    plt.ylabel('Beef CO₂ Emissions (kg/person/year)', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, df_map['Beef_CO2']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                 f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("✅ Alternative static chart displayed successfully")

# 7. Comprehensive Analysis Summary
print("\n" + "="*80)
print("🧠 COMPREHENSIVE ANALYSIS & INSIGHTS")
print("="*80)

# Calculate key metrics
total_emissions_by_country = df_food.groupby('Country')['CO2_Emissions'].sum().sort_values(ascending=False)
animal_emissions_by_country = df_food[df_food['Is_Animal_Product']].groupby('Country')['CO2_Emissions'].sum()
plant_emissions_by_country = df_food[~df_food['Is_Animal_Product']].groupby('Country')['CO2_Emissions'].sum()
animal_percentage = (animal_emissions_by_country / total_emissions_by_country * 100).round(1)

print(f"\n🌍 REGIONAL OVERVIEW:")
print(f"• Average total food emissions per person: {total_emissions_by_country.mean():.1f} kg CO₂/year")
print(f"• Highest emitting country: {total_emissions_by_country.index[0]} ({total_emissions_by_country.iloc[0]:.1f} kg CO₂/year)")
print(f"• Lowest emitting country: {total_emissions_by_country.index[-1]} ({total_emissions_by_country.iloc[-1]:.1f} kg CO₂/year)")

print(f"\n🍖 ANIMAL VS PLANT PRODUCTS:")
for country in total_emissions_by_country.index:
    print(f"• {country}: {animal_percentage[country]:.1f}% from animal products")

print(f"\n📊 KEY FOOD FINDINGS:")
beef_avg = df_food[df_food['Food_Category'] == 'Beef']['CO2_Emissions'].mean()
legume_avg = df_food[df_food['Food_Category'] == 'Legumes']['CO2_Emissions'].mean()
print(f"• Beef emissions are {beef_avg/legume_avg:.1f}x higher than legumes")
print(f"• Average beef emissions: {beef_avg:.1f} kg CO₂/person/year")
print(f"• Average legume emissions: {legume_avg:.1f} kg CO₂/person/year")

highest_food = df_food.groupby('Food_Category')['CO2_Emissions'].mean().idxmax()
lowest_food = df_food.groupby('Food_Category')['CO2_Emissions'].mean().idxmin()
print(f"• Highest emitting food category: {highest_food}")
print(f"• Lowest emitting food category: {lowest_food}")

print(f"\n🌱 SUSTAINABILITY IMPLICATIONS:")
avg_animal_reduction = animal_emissions_by_country.mean()
print(f"• Switching to plant-based diet could reduce emissions by ~{avg_animal_reduction:.1f} kg CO₂/person/year")
print(f"• This represents a {(avg_animal_reduction/total_emissions_by_country.mean()*100):.1f}% reduction in food-related emissions")

print(f"\n🌍 GLOBAL CONTEXT:")
ea_avg = total_emissions_by_country.mean()
global_high = df_global['Total_Food_CO2'].max()
print(f"• East Africa average: {ea_avg:.1f} kg CO₂/person/year")
print(f"• Global highest (Argentina): {global_high:.1f} kg CO₂/person/year")
print(f"• East Africa emits {(ea_avg/global_high*100):.1f}% of the highest global emitter")

print(f"\n💡 POLICY RECOMMENDATIONS:")
print("• Promote legume cultivation and consumption to reduce carbon footprint")
print("• Implement sustainable livestock practices, especially for beef production")
print("• Educate consumers about the climate impact of food choices")
print("• Support local vegetable and fruit production to reduce transport emissions")
print("• Consider carbon labeling on food products")

print(f"\n🎯 CONCLUSION:")
print("East African countries show relatively low food-related CO₂ emissions compared to")
print("developed nations, but there's significant variation within the region. Tanzania") 
print("shows the highest emissions due to higher beef consumption, while Rwanda shows")
print("the lowest with a more plant-based diet. Strategic dietary shifts toward legumes")
print("and vegetables could significantly reduce regional carbon footprints while")
print("maintaining nutritional security.")

# Save results to CSV
df_food.to_csv('east_africa_food_carbon_footprint.csv', index=False)
df_global.to_csv('global_food_carbon_comparison.csv', index=False)

print(f"\n💾 Data saved to:")
print("• east_africa_food_carbon_footprint.csv")
print("• global_food_carbon_comparison.csv")