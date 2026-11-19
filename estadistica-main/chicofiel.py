import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set_theme(style="whitegrid")

# BASE DE DATOS
df = pd.read_csv('base_datos_restaurantes_USA_v2.csv')
df_limpio = df.copy()

print("Base de datos cargada. Filas y columnas originales:", df_limpio.shape)

# ==========================================
# ESPACIO DE ANDRÉS (Estructura Base)
# ==========================================
def limpieza_andres(data):
    # Variables: id_persona, nombre, apellido, genero
    # Como no hay nulos, hacemos limpieza de consistencia (Estandarización de texto)

    # 1. Eliminar espacios en blanco accidentales al inicio o final de las palabras
    data['nombre'] = data['nombre'].str.strip()
    data['apellido'] = data['apellido'].str.strip()
    data['genero'] = data['genero'].str.strip()

    # 2. Formato Título: convierte "juan perez" o "JUAN PEREZ" a "Juan Perez"
    data['nombre'] = data['nombre'].str.title()
    data['apellido'] = data['apellido'].str.title()

    # 3. Capitalizar el género ("masculino" -> "Masculino") para evitar duplicados en los gráficos
    data['genero'] = data['genero'].str.capitalize()

    print("Andrés: Datos de identificación verificados y estandarizados.")
    return data

# ==========================================
# ESPACIO DE MORA (Demográficos)
# ==========================================
def limpieza_mora(data):
    # Variables: edad, ciudad_residencia, estrato_socioeconomico
    # TAREA: Reemplazar nulos de 'edad' con la mediana o media

    # 1) Edades imposibles (-5, 300) -> NaN, para que no distorsionen la mediana
    data.loc[(data['edad'] < 18) | (data['edad'] > 100), 'edad'] = float('nan')

    # 2) Imputar nulos de 'edad' con la mediana (calculada solo con datos válidos)
    mediana_edad = data['edad'].median()
    data['edad'] = data['edad'].fillna(mediana_edad).astype(int)

    # 3) Estandarizar variables categóricas de ubicación y estrato
    data['ciudad_residencia'] = data['ciudad_residencia'].str.strip()
    data['estrato_socioeconomico'] = data['estrato_socioeconomico'].str.strip().str.title()

    print("Mora: Limpieza demográfica aplicada.")
    return data

# ==========================================
# ESPACIO DE NIKO (Consumo Financiero)
# ==========================================
def limpieza_niko(data):
    # Variables: promedio_gasto_comida, frecuencia_visita, ocio

    # 1. Imputamos los valores nulos de 'promedio_gasto_comida' usando la mediana
    mediana_gasto = data['promedio_gasto_comida'].median()
    data['promedio_gasto_comida'] = data['promedio_gasto_comida'].fillna(mediana_gasto)

    # 2. Corregimos valores atípicos/inconsistentes en 'frecuencia_visita' (hay valores negativos de -3)
    # Reemplazamos los valores negativos por la mediana de la frecuencia de visita
    mediana_frecuencia = data['frecuencia_visita'].median()
    data.loc[data['frecuencia_visita'] < 0, 'frecuencia_visita'] = mediana_frecuencia

    print("Niko: Limpieza de consumo financiero aplicada (gasto imputado con mediana y frecuencias corregidas).")
    return data

# ==========================================
# ESPACIO DE TEFA (Fidelización)
# ==========================================
def limpieza_tefa(data):
    # Variables: preferencias_alimenticias, consume_licor, membresia_premium
    # TAREA: Reemplazar nulos de 'preferencias_alimenticias' con la MODA (el valor más común)

    moda_preferencia = data['preferencias_alimenticias'].mode()[0]
    data['preferencias_alimenticias'] = data['preferencias_alimenticias'].fillna(moda_preferencia)

    moda_licor = data['consume_licor'].mode()[0]
    data['consume_licor'] = data['consume_licor'].fillna(moda_licor)

    moda_membresia = data['membresia_premium'].mode()[0]
    data['membresia_premium'] = data['membresia_premium'].fillna(moda_membresia)

    print("Tefa: Limpieza de preferencias aplicada.")
    return data

# ==========================================
# ESPACIO DE CAMILO (Contacto y Pagos)
# ==========================================
def limpieza_camilo(data):
    # Variables: telefono_contacto, correo_electronico, tipo_de_pago_mas_usado, ingresos_mensuales
    # TAREA: Reemplazar nulos masivos con "No registrado" / "Sin registro"

    # 1. Teléfono: unificar formato a XXX-XXX-XXXX
    #    Se quita la extensión (x123), el prefijo de país (+1, 001, 1) y todo lo que no sea dígito
    tel = data['telefono_contacto'].str.strip()
    tel = tel.str.replace(r'[xX].*$', '', regex=True)          # extensión
    tel = tel.str.replace(r'\D', '', regex=True)               # solo dígitos
    tel = tel.str[-10:]                                        # últimos 10 (quita +1 / 001)
    tel = tel.where(tel.str.len() == 10)                       # si no tiene 10 dígitos, se descarta
    tel = tel.str[:3] + '-' + tel.str[3:6] + '-' + tel.str[6:]
    data['telefono_contacto'] = tel

    # 2. Correo: sin espacios y en minúsculas
    data['correo_electronico'] = data['correo_electronico'].str.strip().str.lower()

    # 3. Imputación masiva de medios de contacto
    data['telefono_contacto'] = data['telefono_contacto'].fillna("Sin registro")
    data['correo_electronico'] = data['correo_electronico'].fillna("No registrado")

    # 4. Tipo de pago: consistencia de texto (no tiene nulos)
    data['tipo_de_pago_mas_usado'] = data['tipo_de_pago_mas_usado'].str.strip().str.capitalize()

    # 5. ingresos_mensuales: sin nulos, negativos ni ceros (rango 800 - 17999), no requiere cambios

    print("Camilo: Limpieza de contactos aplicada (formatos unificados y nulos imputados).")
    return data
# ==========================================
# ENSAMBLAJE FINAL
# ==========================================
df_limpio = limpieza_andres(df_limpio)
df_limpio = limpieza_mora(df_limpio)
df_limpio = limpieza_niko(df_limpio)
df_limpio = limpieza_tefa(df_limpio)
df_limpio = limpieza_camilo(df_limpio)

# Verificamos que ya no queden nulos
print("\n--- REVISIÓN FINAL DE DATOS NULOS ---")
print(df_limpio.isnull().sum())
# ==========================================
# RECONOCIMIENTO DE DATOS
# ==========================================
print("\n--- TIPOS DE DATOS ---")
print(df_limpio.dtypes)

print("\n--- ESTADÍSTICOS BÁSICOS (Cuantitativas) ---")
print(df_limpio.describe())

print("\n--- ESTADÍSTICOS BÁSICOS (Cualitativas) ---")
print(df_limpio.describe(include=['object']))

# ==========================================
#  PUNTO 3: ESTADÍSTICOS DE COMPARACIÓN Y GRÁFICOS
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

# Guardar el DataFrame limpio en un nuevo archivo CSV dentro de Colab
df_limpio.to_csv('base_datos_restaurantes_LIMPIA.csv', index=False)

# Configurar el tamaño general de los gráficos
plt.figure(figsize=(15, 10))

# 1. Gráfico Individual: Distribución por Género
plt.subplot(2, 2, 1)
ax1 = sns.countplot(data=df_limpio, x='genero', palette='Set2')
plt.title('Distribución de Clientes por Género')
plt.ylabel('Cantidad')
plt.xlabel('Género')
for p in ax1.patches: # Agregar las etiquetas de datos
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + 0.35, p.get_height() + 100))

# 2. Gráfico Agrupado: Género por Ciudad
plt.subplot(2, 2, 2)
sns.countplot(data=df_limpio, y='ciudad_residencia', hue='genero', palette='Set2')
plt.title('Distribución de Género por Ciudad')
plt.ylabel('Ciudad')
plt.xlabel('Cantidad')

# 3. Distribución Cuantitativa: Histograma de Edades
plt.subplot(2, 2, 3)
sns.histplot(data=df_limpio, x='edad', bins=20, kde=True, color='skyblue')
plt.title('Distribución de Edades de los Clientes')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')

# 4. Gráfico Agrupado: Estrato Socioeconómico y Consumo de Licor
plt.subplot(2, 2, 4)
sns.countplot(data=df_limpio, x='estrato_socioeconomico', hue='consume_licor', palette='pastel')
plt.title('Consumo de Licor según Estrato')
plt.xlabel('Estrato Socioeconómico')
plt.ylabel('Cantidad')

plt.tight_layout()
plt.show()

# ==========================================
#  PUNTO 3.1: BOXPLOT COMPARATIVO Y DIAGRAMA DE TORTA
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 2. Boxplot Comparativo (Cuantitativa: Edad Inicial vs Limpia)
# Se usa un DataFrame con dos columnas de nombres distintos para que seaborn dibuje dos cajas
edad_comparacion = pd.DataFrame({
    'Edad Inicial (con atípicos)': df['edad'],
    'Edad Final (Limpia)': df_limpio['edad']
})
sns.boxplot(
    data=edad_comparacion,
    ax=axes[0, 1],
    palette=['salmon', 'lightgreen']
)
axes[0, 1].set_title("Comparación Boxplot de Edad")

# 3. Diagrama de Torta (Cualitativa: Tipo de Pago)
df_limpio['tipo_de_pago_mas_usado'].value_counts().plot(
    kind='pie', ax=axes[1, 0], autopct='%1.1f%%', startangle=90,
    colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']
)
axes[1, 0].set_title("Distribución por Tipo de Pago")
axes[1, 0].set_ylabel("")

# Ocultar subplots vacíos
axes[0, 0].axis('off')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# ==========================================
#  FASE II: PROBABILIDAD (ejercicios a - j)
#  Probabilidad clásica: P(A) = casos favorables / casos totales
# ==========================================
import os
os.makedirs('markdown', exist_ok=True)   # carpeta donde se guardan los gráficos para el markdown

N = len(df_limpio)
print("\n=== FASE II: PROBABILIDAD === (N =", N, "clientes)")

def etiquetar_barras(ax, formato='{:.1%}'):
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(formato.format(p.get_height()),
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=9)

# Eventos básicos reutilizados en varios puntos
prem  = df_limpio['membresia_premium'] == 'Sí'
licor = df_limpio['consume_licor'] == 'Sí'
alto  = df_limpio['estrato_socioeconomico'] == 'Alto'
carne = df_limpio['preferencias_alimenticias'] == 'Carnes'
veget = df_limpio['preferencias_alimenticias'] == 'Vegetariano'
tarj  = df_limpio['tipo_de_pago_mas_usado'] == 'Tarjeta'
efec  = df_limpio['tipo_de_pago_mas_usado'] == 'Efectivo'

# ------------------------------------------
# a. P(Vegetariano U Carnes) -> eventos excluyentes: P(V) + P(C)
# ------------------------------------------
P_V, P_C = veget.mean(), carne.mean()
P_a = P_V + P_C
print(f"\na) P(V) = {veget.sum()}/{N} = {P_V:.4f} | P(C) = {carne.sum()}/{N} = {P_C:.4f}")
print(f"   P(Vegetariano U Carnes) = {P_V:.4f} + {P_C:.4f} = {P_a:.4f} ({P_a:.2%})")

plt.figure(figsize=(8, 4))
prop = df_limpio['preferencias_alimenticias'].value_counts(normalize=True)
ax = plt.bar(prop.index, prop.values,
             color=['#e8833a' if k in ('Vegetariano', 'Carnes') else '#c9ced6' for k in prop.index])
etiquetar_barras(plt.gca())
plt.title(f'a) Preferencia alimenticia: P(Vegetariano U Carnes) = {P_a:.2%}')
plt.ylabel('Proporción')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_a_preferencia.png', dpi=150)
plt.show()

# ------------------------------------------
# b. P(Premium) = n(Premium) / N
# ------------------------------------------
P_b = prem.mean()
print(f"\nb) P(Premium) = {prem.sum()}/{N} = {P_b:.4f} ({P_b:.2%})")

plt.figure(figsize=(5, 5))
df_limpio['membresia_premium'].value_counts().reindex(['Sí', 'No']).plot(
    kind='pie', autopct='%1.1f%%', startangle=90,
    colors=['#2a6fbb', '#c9ced6'], labels=['Premium', 'Sin Premium'])
plt.ylabel('')
plt.title(f'b) Membresía Premium: P = {P_b:.2%}')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_b_membresia.png', dpi=150)
plt.show()

# ------------------------------------------
# c. P(Licor) = n(Licor) / N
# ------------------------------------------
P_c = licor.mean()
print(f"\nc) P(Licor) = {licor.sum()}/{N} = {P_c:.4f} ({P_c:.2%})")

plt.figure(figsize=(5, 4))
s = df_limpio['consume_licor'].value_counts(normalize=True).reindex(['Sí', 'No'])
plt.bar(['Consume licor', 'No consume'], s.values, color=['#2a6fbb', '#c9ced6'])
etiquetar_barras(plt.gca())
plt.ylim(0, 0.75)
plt.ylabel('Proporción')
plt.title(f'c) Consumo de licor: P = {P_c:.2%}')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_c_licor.png', dpi=150)
plt.show()

# ------------------------------------------
# d. P(Premium | Alto) = n(Premium ∩ Alto) / n(Alto)
# ------------------------------------------
P_d = (prem & alto).sum() / alto.sum()
print(f"\nd) P(Premium | Alto) = {(prem & alto).sum()}/{alto.sum()} = {P_d:.4f} ({P_d:.2%})")

orden_estrato = ['Bajo', 'Medio', 'Alto', 'Muy Alto']
tasa_prem = df_limpio.assign(p=prem).groupby('estrato_socioeconomico')['p'].mean().reindex(orden_estrato)
plt.figure(figsize=(7, 4))
plt.bar(tasa_prem.index, tasa_prem.values, color=['#c9ced6', '#c9ced6', '#e8833a', '#c9ced6'])
plt.axhline(P_b, color='black', linestyle='--', linewidth=1, label=f'P(Premium) global = {P_b:.1%}')
etiquetar_barras(plt.gca())
plt.legend()
plt.ylabel('P(Premium | estrato)')
plt.title('d) Proporción de clientes Premium por estrato')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_d_premium_estrato.png', dpi=150)
plt.show()

# ------------------------------------------
# e. P(Ingresos > 5000 | Licor) = n(I>5000 ∩ Licor) / n(Licor)
# ------------------------------------------
rico = df_limpio['ingresos_mensuales'] > 5000
P_e = (licor & rico).sum() / licor.sum()
print(f"\ne) P(Ingresos>5000 | Licor) = {(licor & rico).sum()}/{licor.sum()} = {P_e:.4f} ({P_e:.2%})")

plt.figure(figsize=(8, 4))
sns.histplot(df_limpio.loc[licor, 'ingresos_mensuales'], bins=40, color='#2a6fbb')
plt.axvline(5000, color='red', linestyle='--')
plt.axvspan(5000, df_limpio['ingresos_mensuales'].max(), color='#e8833a', alpha=0.25)
plt.xlabel('Ingresos mensuales')
plt.title(f'e) Ingresos de quienes consumen licor (zona naranja: P = {P_e:.2%})')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_e_ingresos_licor.png', dpi=150)
plt.show()

# ------------------------------------------
# f. P(Mariscos | Femenino) = n(Mariscos ∩ F) / n(F)
# ------------------------------------------
fem = df_limpio['genero'] == 'Femenino'
mar = df_limpio['preferencias_alimenticias'] == 'Mariscos'
P_f = (mar & fem).sum() / fem.sum()
print(f"\nf) P(Mariscos | Femenino) = {(mar & fem).sum()}/{fem.sum()} = {P_f:.4f} ({P_f:.2%})")

tabla_pref = pd.crosstab(df_limpio['genero'], df_limpio['preferencias_alimenticias'], normalize='index')
tabla_pref.plot(kind='bar', figsize=(9, 4.5), colormap='Set2', rot=0)
plt.ylabel('Proporción dentro de cada género')
plt.title(f'f) Preferencia alimenticia por género: P(Mariscos | F) = {P_f:.2%}')
plt.legend(title='Preferencia', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_f_preferencia_genero.png', dpi=150)
plt.show()

# ------------------------------------------
# g. ¿Licor y Estrato Alto independientes?  P(L ∩ A) =? P(L) * P(A)
# ------------------------------------------
P_LA, P_L, P_A = (licor & alto).mean(), licor.mean(), alto.mean()
print(f"\ng) P(Licor ∩ Alto) = {P_LA:.5f} | P(Licor)*P(Alto) = {P_L:.4f} x {P_A:.4f} = {P_L * P_A:.5f}")
print(f"   P(Licor | Alto) = {P_LA / P_A:.4f} vs P(Licor) = {P_L:.4f}")
chi2_g, p_g, gl_g, _ = stats.chi2_contingency(pd.crosstab(df_limpio['estrato_socioeconomico'], df_limpio['consume_licor']))
print(f"   Chi-cuadrado = {chi2_g:.3f}, gl = {gl_g}, p-valor = {p_g:.4f} ->",
      "se cumple independencia (no se rechaza H0)" if p_g > 0.05 else "NO son independientes (se rechaza H0)")

tasa_licor = df_limpio.assign(l=licor).groupby('estrato_socioeconomico')['l'].mean().reindex(orden_estrato)
plt.figure(figsize=(7, 4))
plt.bar(tasa_licor.index, tasa_licor.values, color=['#c9ced6', '#c9ced6', '#e8833a', '#c9ced6'])
plt.axhline(P_L, color='black', linestyle='--', linewidth=1, label=f'P(Licor) global = {P_L:.1%}')
etiquetar_barras(plt.gca())
plt.ylim(0, 0.8)
plt.legend()
plt.ylabel('P(Licor | estrato)')
plt.title('g) Consumo de licor por estrato')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_g_licor_estrato.png', dpi=150)
plt.show()

# ------------------------------------------
# h. ¿Tarjeta depende de Premium?  P(T | Premium) vs P(T | No Premium) vs P(T)
# ------------------------------------------
P_T, P_T_prem, P_T_no = tarj.mean(), (tarj & prem).sum() / prem.sum(), (tarj & ~prem).sum() / (~prem).sum()
print(f"\nh) P(Tarjeta) = {P_T:.4f} | P(Tarjeta | Premium) = {P_T_prem:.4f} | P(Tarjeta | No Premium) = {P_T_no:.4f}")
chi2_h, p_h, gl_h, _ = stats.chi2_contingency(pd.crosstab(df_limpio['membresia_premium'], tarj))
print(f"   Chi-cuadrado = {chi2_h:.4f}, gl = {gl_h}, p-valor = {p_h:.4f} ->",
      "el uso de tarjeta NO depende de la membresía" if p_h > 0.05 else "el uso de tarjeta SÍ depende de la membresía")

plt.figure(figsize=(6, 4))
plt.bar(['Premium', 'No Premium', 'Global'], [P_T_prem, P_T_no, P_T], color=['#2a6fbb', '#e8833a', '#c9ced6'])
etiquetar_barras(plt.gca())
plt.ylim(0, 0.45)
plt.ylabel('Probabilidad de pagar con tarjeta')
plt.title('h) P(Tarjeta) según membresía')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_h_tarjeta_membresia.png', dpi=150)
plt.show()

# ------------------------------------------
# i. P(Carnes U Efectivo) = P(C) + P(E) - P(C ∩ E)  (no excluyentes)
# ------------------------------------------
P_E, P_CE = efec.mean(), (carne & efec).mean()
P_i = P_C + P_E - P_CE
print(f"\ni) P(C) = {P_C:.4f} | P(E) = {P_E:.4f} | P(C ∩ E) = {P_CE:.4f}")
print(f"   P(Carnes U Efectivo) = {P_C:.4f} + {P_E:.4f} - {P_CE:.4f} = {P_i:.4f} ({P_i:.2%})")

plt.figure(figsize=(7, 4))
plt.bar(['P(Carnes)', 'P(Efectivo)', 'P(C ∩ E)', 'P(C U E)'], [P_C, P_E, P_CE, P_i],
        color=['#c9ced6', '#c9ced6', '#e8833a', '#2a6fbb'])
etiquetar_barras(plt.gca())
plt.ylim(0, 0.7)
plt.title('i) Unión de eventos no excluyentes')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_i_carnes_efectivo.png', dpi=150)
plt.show()

# ------------------------------------------
# j. P(Edad < 30 U Edad > 65) = P(<30) + P(>65)  (excluyentes)
# ------------------------------------------
P_men, P_may = (df_limpio['edad'] < 30).mean(), (df_limpio['edad'] > 65).mean()
P_j = P_men + P_may
print(f"\nj) P(Edad<30) = {P_men:.4f} | P(Edad>65) = {P_may:.4f} | P(<30 U >65) = {P_j:.4f} ({P_j:.2%})")

plt.figure(figsize=(8, 4))
sns.histplot(df_limpio['edad'], bins=range(18, 82), color='#c9ced6')
plt.axvspan(17.5, 29.5, color='#e8833a', alpha=0.4, label=f'< 30: {P_men:.1%}')
plt.axvspan(65.5, 80.5, color='#2a6fbb', alpha=0.4, label=f'> 65: {P_may:.1%}')
plt.legend()
plt.xlabel('Edad')
plt.title(f'j) Segmentos de edad: P total = {P_j:.2%}')
plt.tight_layout()
plt.savefig('markdown/grafico_p2_j_edad.png', dpi=150)
plt.show()

# ------------------------------------------
# RESUMEN FINAL
# ------------------------------------------
resumen_p2 = pd.DataFrame({
    'Ejercicio': list('abcdefghij'),
    'Probabilidad': ['P(V U C)', 'P(Premium)', 'P(Licor)', 'P(Premium | Alto)', 'P(Ingresos>5000 | Licor)',
                     'P(Mariscos | F)', 'P(L ∩ A) vs P(L)P(A)', 'P(Tarjeta | Premium)', 'P(C U E)', 'P(<30 U >65)'],
    'Valor': [P_a, P_b, P_c, P_d, P_e, P_f, P_LA, P_T_prem, P_i, P_j]
})
resumen_p2['Valor (%)'] = (resumen_p2['Valor'] * 100).round(2)
print("\n=== RESUMEN FASE II ===")
print(resumen_p2.to_string(index=False))

# ==========================================================
#  ETAPA 3: MODELOS DE PROBABILIDAD
# ==========================================================
import os
os.makedirs('markdown', exist_ok=True)      # carpeta donde se guardan los gráficos

AZUL, NARANJA, GRIS = '#2a6fbb', '#e8833a', '#9aa3b0'

def formato_p(p):
    """El p-valor puede ser tan pequeño que Python lo redondea a 0.0."""
    return '≈ 0 (< 1e-300)' if p == 0 else f'{p:.3e}'

# ==========================================================
# 1. DISTRIBUCIÓN UNIFORME CONTINUA (EDAD)  ->  X ~ U(18, 80)
# ----------------------------------------------------------
# f(x) = 1 / (b - a)           P(x1 <= X <= x2) = (x2 - x1) / (b - a)
# ==========================================================
print("\n=== 1. DISTRIBUCIÓN UNIFORME (EDAD) ===")
a_edad, b_edad = 18, 80
unif = stats.uniform(loc=a_edad, scale=b_edad - a_edad)      # U(18, 80): loc = a, scale = b - a
print(f"X ~ U({a_edad}, {b_edad}) | f(x) = 1/{b_edad - a_edad} = {1 / (b_edad - a_edad):.5f}")

filas = []
for x1, x2 in [(25, 40), (40, 60), (18, 24)]:
    p_formula = (x2 - x1) / (b_edad - a_edad)                # fórmula (x2 - x1) / (b - a)
    p_scipy = unif.cdf(x2) - unif.cdf(x1)                    # mismo cálculo con scipy
    assert np.isclose(p_formula, p_scipy)                    # ambos métodos deben coincidir
    n_real = df_limpio['edad'].between(x1, x2).sum()         # between() INCLUYE los dos extremos
    p_real = n_real / len(df_limpio)
    # Edades enteras: hay (x2 - x1 + 1) valores posibles de 63 (18..80) -> versión discreta
    p_discreta = (x2 - x1 + 1) / (b_edad - a_edad + 1)
    filas.append([f'[{x1}, {x2}]', f'{x2 - x1}/{b_edad - a_edad}', p_formula, n_real, p_real, p_discreta])

tabla_u = pd.DataFrame(filas, columns=['Intervalo', 'Cálculo', 'P teórica', 'Clientes reales',
                                       'P real', 'P teórica discreta'])
tabla_u['Dif. (real - teórica)'] = tabla_u['P real'] - tabla_u['P teórica']
print(tabla_u.assign(**{c: (tabla_u[c] * 100).round(2).astype(str) + ' %'
                        for c in ['P teórica', 'P real', 'P teórica discreta', 'Dif. (real - teórica)']})
      .to_string(index=False))

# ¿Por qué la real sale un poco MAYOR que la teórica?
# La edad es ENTERA y between() cuenta los dos extremos: en [25, 40] hay 16 edades posibles
# (25, 26, ..., 40), pero el modelo continuo mide el ancho 40 - 25 = 15. Por eso 16/63 > 15/62.
chi2_u, p_chi_u = stats.chisquare(df_limpio['edad'].value_counts().reindex(range(18, 81)).fillna(0))
moda = df_limpio['edad'].mode()[0]
print(f"\nChi-cuadrado de uniformidad (edades 18-80): chi2 = {chi2_u:.1f}, p-valor = {formato_p(p_chi_u)}")
print(f"Edad más frecuente: {moda} años ({(df_limpio['edad'] == moda).sum()} clientes; promedio por edad "
      f"= {len(df_limpio) / 63:.0f}) -> efecto de rellenar con la media/mediana en la limpieza")

# --- Gráfico: densidad real vs. teórica y barras agrupadas teórica vs. real ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_limpio['edad'], bins=np.arange(17.5, 81.5, 1), stat='density', color=GRIS,
             alpha=0.6, edgecolor='white', ax=ax[0], label='Histograma real')
sns.kdeplot(df_limpio['edad'], clip=(18, 80), color=AZUL, linewidth=2, ax=ax[0], label='KDE real')
ax[0].hlines(1 / (b_edad - a_edad), a_edad, b_edad, color=NARANJA, linewidth=2.5,
             label=f'Densidad teórica U(18, 80) = {1 / (b_edad - a_edad):.4f}')
ax[0].set_xlim(15, 83)
ax[0].set_xlabel('Edad (años)')
ax[0].set_ylabel('Densidad')
ax[0].set_title('Edad real vs. densidad uniforme teórica')
ax[0].legend(loc='upper right', fontsize=9)

x, w = np.arange(len(tabla_u)), 0.27
for desplaz, col, color in [(-w, 'P teórica', NARANJA), (0, 'P teórica discreta', GRIS), (w, 'P real', AZUL)]:
    barras = ax[1].bar(x + desplaz, tabla_u[col], width=w, color=color, label=col)
    for r in barras:
        ax[1].annotate(f'{r.get_height():.1%}', (r.get_x() + r.get_width() / 2, r.get_height()),
                       ha='center', va='bottom', fontsize=8.5)
ax[1].set_xticks(x)
ax[1].set_xticklabels(tabla_u['Intervalo'])
ax[1].set_ylim(0, 0.42)
ax[1].set_xlabel('Intervalo de edad')
ax[1].set_ylabel('Probabilidad')
ax[1].set_title('Probabilidad teórica vs. real por intervalo')
ax[1].legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('markdown/grafico_p3_a_uniforme.png', dpi=150)
plt.show()

# ==========================================================
# 2. DISTRIBUCIÓN DE POISSON (FRECUENCIA DE VISITA)  ->  X ~ Poisson(λ)
# ----------------------------------------------------------
# P(X = k) = e^(-λ) · λ^k / k!        P(X > 8) = 1 - P(X <= 8)  (regla del complemento)
# λ = media muestral. Se asume que `frecuencia_visita` son visitas por mes.
# ==========================================================
print("\n=== 2. DISTRIBUCIÓN DE POISSON (FRECUENCIA DE VISITA) ===")
visitas = df_limpio['frecuencia_visita']
lam = visitas.mean()
print(f"λ = media muestral = {lam:.4f} | varianza = {visitas.var():.4f} | "
      f"índice de dispersión var/media = {visitas.var() / lam:.3f} (Poisson ideal = 1)")

p2_teo = stats.poisson.pmf(2, lam)                    # P(X = 2) con la PMF
p_hasta8 = stats.poisson.cdf(8, lam)                  # P(X <= 8) con la CDF
p8_teo = 1 - p_hasta8                                 # P(X > 8) = 1 - P(X <= 8)
p2_real = (visitas == 2).mean()
p8_real = (visitas > 8).mean()
print(f"P(X <= 8) = {p_hasta8:.4f}  ->  P(X > 8) = 1 - {p_hasta8:.4f} = {p8_teo:.4f}")

tabla_p = pd.DataFrame({'Evento': ['P(X = 2)', 'P(X > 8)'],
                        'P teórica (Poisson)': [p2_teo, p8_teo],
                        'Clientes reales': [(visitas == 2).sum(), (visitas > 8).sum()],
                        'P real': [p2_real, p8_real]})
tabla_p['Dif. (real - teórica)'] = tabla_p['P real'] - tabla_p['P teórica (Poisson)']
print(tabla_p.assign(**{c: (tabla_p[c] * 100).round(2).astype(str) + ' %'
                        for c in ['P teórica (Poisson)', 'P real', 'Dif. (real - teórica)']}).to_string(index=False))

# Distribución completa k = 0..10 y bondad de ajuste (última clase agrupa k >= 10; 1 parámetro estimado)
ks = np.arange(0, 11)
frec_real = np.array([(visitas == k).mean() for k in ks])
frec_teo = stats.poisson.pmf(ks, lam)
esperado = frec_teo * len(visitas)
esperado[-1] += stats.poisson.sf(10, lam) * len(visitas)
observado = frec_real * len(visitas)
chi2_p, p_chi_p = stats.chisquare(observado, esperado * observado.sum() / esperado.sum(), ddof=1)
print(f"Bondad de ajuste: chi2 = {chi2_p:.1f}, gl = {len(ks) - 2}, p-valor = {formato_p(p_chi_p)}")
ceros = (visitas == 0).sum()
print(f"Clientes con 0 visitas: {ceros} ({ceros / len(visitas):.2%}) vs modelo {frec_teo[0]:.2%}"
      f" | de ellos con gasto = 0: {((visitas == 0) & (df_limpio['promedio_gasto_comida'] == 0)).sum()}")

# --- Gráfico: modelo vs. datos observados ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.6, 1]})
w = 0.4
ax[0].bar(ks - w / 2, frec_real, width=w, color=AZUL, label='Real (dataset)')
ax[0].bar(ks + w / 2, frec_teo, width=w, color=NARANJA, label=f'Poisson (λ = {lam:.2f})')
ax[0].axvspan(8.5, 10.6, color=GRIS, alpha=0.2)
ax[0].text(9.55, 0.10, 'Zona\nX > 8', ha='center', fontsize=9.5)
ax[0].set_xticks(ks)
ax[0].set_xlabel('Visitas en un mes (k)')
ax[0].set_ylabel('Probabilidad / frecuencia relativa')
ax[0].set_title('Poisson teórica vs. frecuencia de visitas observada')
ax[0].legend(loc='upper left', fontsize=9)

x, w = np.arange(2), 0.35
for desplaz, col, color in [(-w / 2, 'P real', AZUL), (w / 2, 'P teórica (Poisson)', NARANJA)]:
    barras = ax[1].bar(x + desplaz, tabla_p[col], width=w, color=color, label=col)
    for r in barras:
        ax[1].annotate(f'{r.get_height():.2%}', (r.get_x() + r.get_width() / 2, r.get_height()),
                       ha='center', va='bottom', fontsize=9)
ax[1].set_xticks(x)
ax[1].set_xticklabels(tabla_p['Evento'])
ax[1].set_ylim(0, 0.16)
ax[1].set_ylabel('Probabilidad')
ax[1].set_title('Las dos probabilidades pedidas')
ax[1].legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('markdown/grafico_p3_b_poisson.png', dpi=150)
plt.show()

# ==========================================================
# 3. DISTRIBUCIÓN NORMAL, TLC Y PRUEBA DE HIPÓTESIS (GASTO EN COMIDA)
# ==========================================================
print("\n=== 3. TLC Y PRUEBA DE HIPÓTESIS: GASTO PREMIUM vs NO PREMIUM ===")

# ---- 3.1 Hipótesis (prueba unilateral derecha) ----
alfa = 0.05
print("H0: μ_Premium - μ_NoPremium <= 0   (el programa no aumenta el gasto)")
print("H1: μ_Premium - μ_NoPremium  > 0   (los Premium gastan más)")
print(f"Nivel de significancia α = {alfa}")

# ---- 3.2 Datos muestrales y supuestos ----
g_prem = df_limpio.loc[df_limpio['membresia_premium'] == 'Sí', 'promedio_gasto_comida']
g_no = df_limpio.loc[df_limpio['membresia_premium'] == 'No', 'promedio_gasto_comida']
n1, n2 = len(g_prem), len(g_no)
m1, m2 = g_prem.mean(), g_no.mean()
v1, v2 = g_prem.var(ddof=1), g_no.var(ddof=1)
print(pd.DataFrame({'n': [n1, n2], 'Media': [m1, m2], 'Desv. estándar': [g_prem.std(), g_no.std()],
                    'Varianza': [v1, v2], 'Asimetría': [g_prem.skew(), g_no.skew()]},
                   index=['Premium', 'No Premium']).round(2).to_string())
lev_stat, lev_p = stats.levene(g_prem, g_no)
print(f"Razón de varianzas Premium/No Premium = {v1 / v2:.2f} | Levene: p-valor = {formato_p(lev_p)} "
      f"-> varianzas {'DISTINTAS' if lev_p < alfa else 'iguales'}")

# ---- 3.3 Demostración visual del TLC: 1000 muestras de n = 50 por grupo ----
rng = np.random.default_rng(42)                                    # semilla fija = resultados reproducibles
n_muestras, tam = 1000, 50
medias_prem = rng.choice(g_prem.values, size=(n_muestras, tam)).mean(axis=1)
medias_no = rng.choice(g_no.values, size=(n_muestras, tam)).mean(axis=1)
print(f"\nTLC (1000 muestras de n = 50):")
print(f"  Asimetría datos individuales: Premium {g_prem.skew():.2f}, No Premium {g_no.skew():.2f}")
print(f"  Asimetría de las medias muestrales: Premium {stats.skew(medias_prem):.2f}, No Premium {stats.skew(medias_no):.2f}")
print(f"  Error estándar simulado vs. teórico σ/√n: Premium {medias_prem.std():.2f} vs {g_prem.std() / np.sqrt(tam):.2f} | "
      f"No Premium {medias_no.std():.2f} vs {g_no.std() / np.sqrt(tam):.2f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for j, (datos, medias, color, nombre) in enumerate([(g_prem, medias_prem, AZUL, 'Premium'),
                                                    (g_no, medias_no, NARANJA, 'No Premium')]):
    sns.histplot(datos, bins=40, stat='density', color=color, alpha=0.7, edgecolor='white', ax=ax[0, j])
    ax[0, j].set_title(f'{nombre}: gasto individual (asimetría = {datos.skew():.2f})')
    ax[0, j].set_xlabel('Gasto promedio por visita ($)')
    ax[0, j].set_ylabel('Densidad')
    sns.histplot(medias, bins=30, stat='density', color=color, alpha=0.7, edgecolor='white', ax=ax[1, j])
    xs = np.linspace(medias.min(), medias.max(), 300)
    ax[1, j].plot(xs, stats.norm.pdf(xs, datos.mean(), datos.std() / np.sqrt(tam)), color='black',
                  linewidth=2, label='Normal teórica N(μ, σ/√n)')
    ax[1, j].set_title(f'{nombre}: medias de 1000 muestras (n = 50)')
    ax[1, j].set_xlabel('Media muestral del gasto ($)')
    ax[1, j].set_ylabel('Densidad')
    ax[1, j].legend(fontsize=9)
plt.tight_layout()
plt.savefig('markdown/grafico_p3_c1_tlc.png', dpi=150)
plt.show()

# ---- 3.4 Estadísticos de prueba: Welch, Student y Z ----
# Welch:   t = (x1 - x2) / sqrt(s1²/n1 + s2²/n2),  gl = (s1²/n1 + s2²/n2)² / [(s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1)]
# Student: usa la varianza combinada sp²,           gl = n1 + n2 - 2
# Z:       mismo cociente que Welch pero con la N(0,1) como distribución de referencia
welch = stats.ttest_ind(g_prem, g_no, equal_var=False, alternative='greater')
student = stats.ttest_ind(g_prem, g_no, equal_var=True, alternative='greater')
se_welch = np.sqrt(v1 / n1 + v2 / n2)
gl_welch = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
gl_student = n1 + n2 - 2
z_obs = (m1 - m2) / se_welch
p_z = stats.norm.sf(z_obs)

resumen_ht = pd.DataFrame({
    'Prueba': ['t de Welch', 't de Student', 'Normal Z'],
    'Estadístico': [welch.statistic, student.statistic, z_obs],
    'Grados de libertad': [gl_welch, gl_student, np.nan],
    'Valor crítico (α=0.05)': [stats.t.ppf(1 - alfa, gl_welch), stats.t.ppf(1 - alfa, gl_student), stats.norm.ppf(1 - alfa)],
    'p-valor': [formato_p(welch.pvalue), formato_p(student.pvalue), formato_p(p_z)],
    'Decisión': ['Rechaza H0' if p < alfa else 'No rechaza H0' for p in (welch.pvalue, student.pvalue, p_z)]})
print("\n", resumen_ht.round(3).to_string(index=False))

# ---- 3.5 Tamaño del efecto e intervalo de confianza al 95 % ----
dif = m1 - m2
ic = (dif - stats.t.ppf(0.975, gl_welch) * se_welch, dif + stats.t.ppf(0.975, gl_welch) * se_welch)
print(f"\nDiferencia de medias = ${dif:.2f} | IC 95 % = (${ic[0]:.2f}, ${ic[1]:.2f}) | "
      f"d de Cohen = {dif / np.sqrt((v1 + v2) / 2):.2f} | Premium gasta {dif / m2:.0%} más")
mw = stats.mannwhitneyu(g_prem, g_no, alternative='greater')
print(f"Verificación sin supuesto de normalidad (Mann-Whitney): p-valor = {formato_p(mw.pvalue)}")

# Región de rechazo
zz = np.linspace(-4, 6, 600)
zc = stats.norm.ppf(1 - alfa)
plt.figure(figsize=(7, 4.5))
plt.plot(zz, stats.norm.pdf(zz), color='black', linewidth=1.5)
plt.fill_between(zz[zz >= zc], stats.norm.pdf(zz[zz >= zc]), color=NARANJA, alpha=0.6, label=f'Región de rechazo (α = {alfa})')
plt.axvline(zc, color=NARANJA, linestyle='--')
plt.annotate(f'Estadístico observado\nt = {welch.statistic:.1f} (fuera del gráfico)', xy=(5.9, 0.01),
             xytext=(2.1, 0.25), fontsize=9.5, arrowprops=dict(arrowstyle='->', color=AZUL, lw=2))
plt.text(zc + 0.15, 0.15, f'valor crítico\n= {zc:.3f}', fontsize=9, color=NARANJA)
plt.xlabel('Estadístico de prueba (escala estándar)')
plt.ylabel('Densidad')
plt.title('Decisión: prueba de cola derecha, α = 0.05')
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('markdown/grafico_p3_c2_region_rechazo.png', dpi=150)
plt.show()

# ---- 3.6 Conexión con la Etapa 2: confusión por estrato socioeconómico ----
orden_est = ['Bajo', 'Medio', 'Alto', 'Muy Alto']
es_prem = df_limpio['membresia_premium'] == 'Sí'
pct_alto = df_limpio.loc[es_prem, 'estrato_socioeconomico'].isin(['Alto', 'Muy Alto']).mean()
print(f"\nCONFUSIÓN POR ESTRATO: {pct_alto:.2%} de los clientes Premium son de estrato Alto o Muy Alto")

filas_e, pesos, difs, varzs = [], [], [], []
for est in orden_est:
    sub = df_limpio[df_limpio['estrato_socioeconomico'] == est]
    ga = sub.loc[sub['membresia_premium'] == 'Sí', 'promedio_gasto_comida']
    gb = sub.loc[sub['membresia_premium'] == 'No', 'promedio_gasto_comida']
    r = stats.ttest_ind(ga, gb, equal_var=False, alternative='greater')
    filas_e.append([est, len(ga), len(gb), ga.mean(), gb.mean(), ga.mean() - gb.mean(), formato_p(r.pvalue)])
    pesos.append(len(sub) / len(df_limpio))
    difs.append(ga.mean() - gb.mean())
    varzs.append(ga.var(ddof=1) / len(ga) + gb.var(ddof=1) / len(gb))
tabla_e = pd.DataFrame(filas_e, columns=['Estrato', 'n Premium', 'n No Premium', 'Gasto Premium',
                                         'Gasto No Premium', 'Diferencia', 'p-valor (cola der.)'])
print(tabla_e.round(2).to_string(index=False))

# Diferencia ajustada por estrato = promedio de las diferencias dentro de cada estrato (peso = tamaño del estrato)
pesos, difs, varzs = np.array(pesos), np.array(difs), np.array(varzs)
dif_aj = (pesos * difs).sum()
se_aj = np.sqrt((pesos ** 2 * varzs).sum())
ic_aj = (dif_aj - 1.96 * se_aj, dif_aj + 1.96 * se_aj)
p_aj = stats.norm.sf(dif_aj / se_aj)
print(f"\nDiferencia CRUDA = ${dif:.2f}  |  diferencia AJUSTADA por estrato = ${dif_aj:.2f} "
      f"(IC 95 %: ${ic_aj[0]:.2f} a ${ic_aj[1]:.2f}, p-valor cola der. = {p_aj:.3f})")
print(f"El estrato explica ≈ {1 - dif_aj / dif:.0%} de la diferencia cruda de gasto")

fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.3, 1]})
x, w = np.arange(4), 0.38
for desplaz, col, color, nombre in [(-w / 2, 'Gasto Premium', AZUL, 'Premium'), (w / 2, 'Gasto No Premium', NARANJA, 'No Premium')]:
    barras = ax[0].bar(x + desplaz, tabla_e[col], width=w, color=color, label=nombre)
    for r in barras:
        ax[0].annotate(f'${r.get_height():.1f}', (r.get_x() + r.get_width() / 2, r.get_height()),
                       ha='center', va='bottom', fontsize=9)
ax[0].set_xticks(x)
ax[0].set_xticklabels(orden_est)
ax[0].set_xlabel('Estrato socioeconómico')
ax[0].set_ylabel('Gasto promedio por visita ($)')
ax[0].set_title('Dentro de cada estrato, Premium y No Premium gastan casi lo mismo')
ax[0].legend(fontsize=9)

ax[1].errorbar([dif, dif_aj], [1, 0], xerr=[[dif - ic[0], dif_aj - ic_aj[0]], [ic[1] - dif, ic_aj[1] - dif_aj]],
               fmt='o', color='black', capsize=6, markersize=8)
ax[1].axvline(0, color='gray', linestyle=':')
ax[1].text(dif, 1.12, f'${dif:.2f}', ha='center', fontsize=10)
ax[1].text(dif_aj + 1.5, 0.12, f'${dif_aj:.2f}', ha='left', fontsize=10)
ax[1].set_yticks([0, 1])
ax[1].set_yticklabels(['Ajustada\npor estrato', 'Cruda\n(sin ajustar)'])
ax[1].set_ylim(-0.5, 1.5)
ax[1].set_xlabel('Diferencia de gasto Premium - No Premium ($) con IC 95 %')
ax[1].set_title('La diferencia casi desaparece al controlar por estrato')
plt.tight_layout()
plt.savefig('markdown/grafico_p3_c3_confusion_estrato.png', dpi=150)
plt.show()

# ---- 3.7 Conclusión ----
print("\nCONCLUSIÓN:")
print(f"  p-valor {formato_p(welch.pvalue)}, menor que α = {alfa} -> se RECHAZA H0: los Premium (${m1:.2f}) gastan en promedio "
      f"más que los No Premium (${m2:.2f}), una diferencia de ${dif:.2f} (IC 95 %: ${ic[0]:.2f} a ${ic[1]:.2f}).")
print(f"  Pero {pct_alto:.0%} de los Premium son de estrato Alto/Muy Alto y el estrato determina el gasto: "
      f"al ajustar por estrato la diferencia baja a ${dif_aj:.2f}. La asociación es real, "
      f"pero no demuestra que la membresía CAUSE el mayor gasto.")

# Sensibilidad (solo si la base original está en `df`): ¿cambia algo sin los gastos imputados?
if 'df' in globals():
    orig_ok = df.dropna(subset=['promedio_gasto_comida'])
    t_sin = stats.ttest_ind(orig_ok.loc[orig_ok['membresia_premium'] == 'Sí', 'promedio_gasto_comida'],
                            orig_ok.loc[orig_ok['membresia_premium'] == 'No', 'promedio_gasto_comida'],
                            equal_var=False, alternative='greater')
    print(f"\nSensibilidad (sin los {len(df) - len(orig_ok)} gastos imputados): t de Welch = {t_sin.statistic:.2f}, "
          f"p-valor = {formato_p(t_sin.pvalue)} -> la conclusión no cambia")