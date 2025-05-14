import pandas as pd

# 1. Leer el archivo CSV
df = pd.read_csv('titanic.csv')

# 2. Información básica
print("Dimensiones del DataFrame:")
print(df.shape)

print("\nNúmero de datos que contiene:")
print(df.size)

print("\nNombres de las columnas:")
print(df.columns.tolist())

print("\nTipos de datos de las columnas:")
print(df.dtypes)

print("\nPrimeras 10 filas:")
print(df.head(10))

print("\nÚltimas 10 filas:")
print(df.tail(10))

# 3. Datos del pasajero con ID 148
print("\nDatos del pasajero con ID 148:")
print(df[df['PassengerId'] == 148])

# 4. Filas pares
print("\nFilas pares del DataFrame:")
print(df.iloc[::2])

# 5. Nombres de personas en primera clase ordenados alfabéticamente
print("\nNombres de personas en primera clase ordenados alfabéticamente:")
first_class_names = df[df['Pclass'] == 1]['Name'].sort_values()
print(first_class_names)

# 6. Porcentaje de personas que sobrevivieron y murieron
print("\nPorcentaje de personas que sobrevivieron y murieron:")
survival_percent = df['Survived'].value_counts(normalize=True) * 100
print(survival_percent.rename({0: 'Murieron', 1: 'Sobrevivieron'}))

# 7. Porcentaje de personas que sobrevivieron en cada clase
print("\nPorcentaje de personas que sobrevivieron en cada clase:")
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
print(class_survival.rename({1: 'Primera', 2: 'Segunda', 3: 'Tercera'}))

# 8. Eliminar pasajeros con edad desconocida
print("\nDataFrame después de eliminar pasajeros con edad desconocida:")
df_clean = df.dropna(subset=['Age'])
print(df_clean)

# 9. Edad media de las mujeres en cada clase
print("\nEdad media de las mujeres en cada clase:")
women_age = df_clean[df_clean['Sex'] == 'female'].groupby('Pclass')['Age'].mean()
print(women_age.rename({1: 'Primera', 2: 'Segunda', 3: 'Tercera'}))

# 10. Añadir columna booleana 'Minor'
print("\nDataFrame con columna 'Minor' (menor de edad):")
df_clean['Minor'] = df_clean['Age'] < 18
print(df_clean)

# 11. Porcentaje de menores y mayores que sobrevivieron en cada clase
print("\nPorcentaje de menores y mayores que sobrevivieron en cada clase:")
minor_survival = df_clean.groupby(['Pclass', 'Minor'])['Survived'].mean() * 100
print(minor_survival.unstack().rename(columns={False: 'Mayores', True: 'Menores'},
                                      index={1: 'Primera', 2: 'Segunda', 3: 'Tercera'}))
