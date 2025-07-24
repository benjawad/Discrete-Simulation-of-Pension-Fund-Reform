import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from flask import Flask, request, render_template, send_from_directory, jsonify, send_file
from flask_cors import CORS
from datetime import datetime

# Gestion de FPDF
PDF_AVAILABLE = True
try:
    from fpdf import FPDF
except ImportError:
    PDF_AVAILABLE = False
    print("fpdf non disponible. La génération PDF sera désactivée.")

# Installation de numba (optionnelle)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    print("Numba non disponible. Utilisation de la version standard.")
    NUMBA_AVAILABLE = False
    # Définir un décorateur dummy pour jit
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

# --- Configuration Initiale ---
INITIAL_RESERVE = 200000000
INITIAL_EMPLOYEES_COUNT = 10000
INITIAL_RETIREES_COUNT = 1000
SIMULATION_START_YEAR = 2025
SIMULATION_END_YEAR = 2035
NUM_SIMULATIONS = 40

CURRENT_RETIREMENT_AGE = 63

# Distributions converties en numpy arrays pour plus d'efficacité
SALARY_DIST_INITIAL = np.array([
    (3000, 5000, 0.20), (5000, 7500, 0.40), (7500, 10000, 0.60),
    (10000, 15000, 0.80), (15000, 20000, 0.90), (20000, 30000, 0.95),
    (30000, 40000, 1.00)
])
AGE_DIST_INITIAL = np.array([
    (21, 30, 0.20), (31, 40, 0.50), (41, 52, 0.80), (53, 63, 1.00)
])
AGE_DIST_HIRING = np.array([
    (21, 24, 0.05), (25, 28, 0.35), (29, 32, 0.65), (33, 36, 0.80),
    (37, 40, 0.95), (41, 45, 1.00)
])
SALARY_DIST_HIRING = np.array([
    (3000, 4000, 0.20), (4000, 6000, 0.40), (6000, 8000, 0.60),
    (8000, 12000, 0.80), (12000, 16000, 0.90), (16000, 24000, 0.95),
    (24000, 32000, 1.00)
])
CONTRIBUTION_RATES_S1_S2 = np.array([
    (0, 4999.99, 0.05), (5000, 6999.99, 0.06), (7000, 9999.99, 0.08),
    (10000, float('inf'), 0.10)
])
CONTRIBUTION_RATES_S3_S4 = np.array([
    (0, 4999.99, 0.06), (5000, 6999.99, 0.08), (7000, 9999.99, 0.10),
    (10000, float('inf'), 0.14)
])

SCENARIOS_PARAMS = {
    1: ("Scenario 1: Retraite 63 ans", CURRENT_RETIREMENT_AGE, CONTRIBUTION_RATES_S1_S2, 2.0/100),
    2: ("Scenario 2: Retraite 65 ans", 65, CONTRIBUTION_RATES_S1_S2, 2.0/100),
    3: ("Scenario 3: Retraite 65 ans, cotisations augmentées", 65, CONTRIBUTION_RATES_S3_S4, 2.0/100),
    4: ("Scenario 4: Retraite 65 ans, cotis+, pension-", 65, CONTRIBUTION_RATES_S3_S4, 1.5/100)
}

# --- Fonction Alea optimisée avec Numba ---
@jit(nopython=True)
def alea_optimized(IX, IY, IZ):
    """Version optimisée avec Numba du générateur de nombres pseudo-aléatoires"""
    IX = 171 * (IX % 177) - 2 * (IX // 177)
    IY = 172 * (IY % 176) - 35 * (IY // 176)
    IZ = 170 * (IZ % 178) - 63 * (IZ // 178)
    
    if IX < 0: IX += 30269
    if IY < 0: IY += 30307
    if IZ < 0: IZ += 30323
    
    inter = (IX / 30269.0) + (IY / 30307.0) + (IZ / 30323.0)
    return inter - int(inter), IX, IY, IZ

@jit(nopython=True)
def get_random_value_from_distribution_optimized(dist_table, rand_num):
    """Version optimisée pour obtenir une valeur aléatoire d'une distribution"""
    for i in range(len(dist_table)):
        min_val, max_val, freq_cumul = dist_table[i]
        if rand_num <= freq_cumul:
            return min_val + rand_num * (max_val - min_val)
    return dist_table[-1][1]  # Retourne la valeur max si rien n'est trouvé

@jit(nopython=True)
def get_contribution_rate_optimized(salary, rates_table):
    """Version optimisée pour obtenir le taux de cotisation"""
    for i in range(len(rates_table)):
        min_sal, max_sal, rate = rates_table[i]
        if min_sal <= salary <= max_sal:
            return rate
    return 0.0

# --- Classes optimisées avec des structures numpy ---
class OptimizedEmployee:
    """Classe employé optimisée avec des attributs essentiels uniquement"""
    __slots__ = ['age', 'salary', 'years_worked', 'hire_year']
    
    def __init__(self, age, salary, years_worked, hire_year):
        self.age = age
        self.salary = salary
        self.years_worked = years_worked
        self.hire_year = hire_year

class OptimizedRetiree:
    """Classe retraité optimisée"""
    __slots__ = ['pension', 'retirement_year', 'age_at_retirement', 'current_age']
    
    def __init__(self, pension, retirement_year, age_at_retirement):
        self.pension = pension
        self.retirement_year = retirement_year
        self.age_at_retirement = age_at_retirement
        self.current_age = age_at_retirement

# --- Initialisation optimisée ---
def initialize_population_optimized(start_year, ix, iy, iz):
    """Version optimisée de l'initialisation de la population"""
    employees = []
    retirees = []
    
    # Pré-générer tous les nombres aléatoires nécessaires
    current_ix, current_iy, current_iz = ix, iy, iz
    
    # Initialiser les employés
    for _ in range(INITIAL_EMPLOYEES_COUNT):
        # Générer âge
        rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
        age = int(get_random_value_from_distribution_optimized(AGE_DIST_INITIAL, rand_val))
        
        # Générer salaire
        rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
        salary = get_random_value_from_distribution_optimized(SALARY_DIST_INITIAL, rand_val)
        
        # Générer âge à l'embauche
        rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
        age_at_hiring = int(get_random_value_from_distribution_optimized(AGE_DIST_HIRING, rand_val))
        
        # Ajustements logiques
        if age_at_hiring >= age:
            age_at_hiring = max(21, age - 1)
        
        years_worked = max(0, age - age_at_hiring)
        hire_year = start_year - years_worked
        
        employees.append(OptimizedEmployee(age, salary, years_worked, hire_year))
    
    # Initialiser les retraités
    pension_factor = 2.0 / 100.0
    for _ in range(INITIAL_RETIREES_COUNT):
        rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
        dsar = get_random_value_from_distribution_optimized(SALARY_DIST_INITIAL, rand_val)
        
        rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
        age_at_hiring = int(get_random_value_from_distribution_optimized(AGE_DIST_HIRING, rand_val))
        
        nat = max(15, CURRENT_RETIREMENT_AGE)
        pension = nat * pension_factor * dsar
        
        retirees.append(OptimizedRetiree(pension, start_year - 1, CURRENT_RETIREMENT_AGE))
    
    return employees, retirees, (current_ix, current_iy, current_iz)

# --- Simulation annuelle optimisée ---
def run_annual_step_optimized(current_year, employees, retirees, reserve, scenario_params, seeds):
    """Version optimisée de la simulation annuelle"""
    retirement_age, contribution_table, pension_nat_coeff = scenario_params[1], scenario_params[2], scenario_params[3]
    current_ix, current_iy, current_iz = seeds
    
    # Incrémenter l'âge des retraités
    for retiree in retirees:
        retiree.current_age += 1
    
    # Augmentation salariale conditionnelle
    salary_increase = 1.05 if current_year in [2025, 2030, 2035] else 1.0
    
    # Traitement des employés
    newly_retired = []
    remaining_employees = []
    
    for emp in employees:
        emp.age += 1
        emp.years_worked += 1
        emp.salary *= salary_increase
        
        if emp.age >= retirement_age:
            pension_amount = emp.years_worked * pension_nat_coeff * emp.salary
            newly_retired.append(OptimizedRetiree(pension_amount, current_year, emp.age))
        else:
            remaining_employees.append(emp)
    
    employees = remaining_employees
    retirees.extend(newly_retired)
    
    # Nouveaux recrutements
    rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
    num_new_recruits = int(250 + rand_val * 150)  # Entre 250 et 400
    
    for _ in range(num_new_recruits):
        rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
        age = int(get_random_value_from_distribution_optimized(AGE_DIST_HIRING, rand_val))
        
        rand_val, current_ix, current_iy, current_iz = alea_optimized(current_ix, current_iy, current_iz)
        salary = get_random_value_from_distribution_optimized(SALARY_DIST_HIRING, rand_val)
        
        employees.append(OptimizedEmployee(age, salary, 0, current_year))
    
    # Calculs financiers
    total_cotis = sum(emp.salary * get_contribution_rate_optimized(emp.salary, contribution_table) * 12 
                     for emp in employees)
    total_pens = sum(ret.pension * 12 for ret in retirees)
    
    reserve += total_cotis - total_pens
    
    indicators = {
        "Year": current_year,
        "TotEmp": len(employees),
        "TotRet": len(retirees),
        "TotCotis": total_cotis,
        "TotPens": total_pens,
        "Reserve": reserve,
        "NouvRet": len(newly_retired),
        "NouvRec": num_new_recruits
    }
    
    return employees, retirees, reserve, indicators, (current_ix, current_iy, current_iz)

# --- Simulation d'un scénario optimisée ---
def run_single_scenario_simulation_optimized(scenario_id, initial_seeds):
    """Version optimisée de la simulation d'un scénario"""
    ix, iy, iz = initial_seeds
    
    # Initialisation
    employees, retirees, seeds = initialize_population_optimized(SIMULATION_START_YEAR, ix, iy, iz)
    current_reserve = INITIAL_RESERVE
    
    scenario_params = SCENARIOS_PARAMS[scenario_id]
    annual_results = []
    
    for year in range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1):
        employees, retirees, current_reserve, indicators, seeds = \
            run_annual_step_optimized(year, employees, retirees, current_reserve, scenario_params, seeds)
        annual_results.append(indicators)
    
    return annual_results

# --- Classe PDF personnalisée ---
class SimulationPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Rapport de Simulation - Système de Retraite', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_table(self, headers, data, title=None):
        if title:
            self.chapter_title(title)
        
        # En-têtes
        self.set_font('Arial', 'B', 9)
        col_width = 190 / len(headers)
        
        for header in headers:
            self.cell(col_width, 7, str(header), 1, 0, 'C')
        self.ln()
        
        # Données
        self.set_font('Arial', '', 8)
        for row in data:
            for item in row:
                if isinstance(item, float):
                    self.cell(col_width, 6, f"{item:,.0f}", 1, 0, 'R')
                else:
                    self.cell(col_width, 6, str(item), 1, 0, 'C')
            self.ln()
        self.ln(5)

def generate_comprehensive_pdf_report(results_data, confidence_intervals, overall_reserve_table):
    """Génère un rapport PDF complet"""
    if not PDF_AVAILABLE:
        return None
    
    try:
        pdf = SimulationPDF()
        pdf.add_page()
        
        # Introduction
        intro_text = f"""Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}

Ce rapport présente les résultats de la simulation actuarielle du système de retraite sur la période {SIMULATION_START_YEAR}-{SIMULATION_END_YEAR}.

Paramètres de simulation:
- Nombre de simulations par scénario: {NUM_SIMULATIONS}
- Réserve initiale: {INITIAL_RESERVE:,} DH
- Nombre d'employés initial: {INITIAL_EMPLOYEES_COUNT:,}
- Nombre de retraités initial: {INITIAL_RETIREES_COUNT:,}
"""
        pdf.chapter_body(intro_text)
        
        # Scénarios analysés
        pdf.chapter_title("Scénarios analysés:")
        scenarios_text = ""
        for scenario_id, data in results_data.items():
            scenarios_text += f"- {data['name']}\n"
        pdf.chapter_body(scenarios_text)
        
        # Tableau des réserves moyennes
        pdf.add_page()
        pdf.chapter_title("Évolution des réserves moyennes (en DH)")
        
        headers = ["Année"] + [results_data[sid]['name'] for sid in results_data.keys()]
        table_data = []
        
        for row in overall_reserve_table:
            table_row = [row['Year']]
            for scenario_id in results_data.keys():
                scenario_name = results_data[scenario_id]['name']
                if scenario_name in row:
                    table_row.append(row[scenario_name])
                else:
                    table_row.append(0)
            table_data.append(table_row)
        
        pdf.add_table(headers, table_data)
        
        # Intervalles de confiance
        pdf.chapter_title("Intervalles de confiance (95%) pour les réserves")
        
        for year in [2025, 2030, 2035]:
            pdf.chapter_title(f"Année {year}")
            ci_headers = ["Scénario", "Moyenne", "IC Inférieur", "IC Supérieur", "Écart-type"]
            ci_data = []
            
            for scenario_id in results_data.keys():
                if scenario_id in confidence_intervals and year in confidence_intervals[scenario_id]:
                    ci_info = confidence_intervals[scenario_id][year]
                    ci_data.append([
                        results_data[scenario_id]['name'],
                        ci_info['Moyenne'],
                        ci_info['IC_inf'],
                        ci_info['IC_sup'],
                        ci_info['Ecart-type']
                    ])
            
            if ci_data:
                pdf.add_table(ci_headers, ci_data)
        
        # Analyse détaillée par scénario
        pdf.add_page()
        pdf.chapter_title("Analyse détaillée par scénario")
        
        for scenario_id, data in results_data.items():
            pdf.chapter_title(f"{data['name']}")
            
            # Données moyennes
            avg_data = data['averages']
            if avg_data:
                headers = ["Année", "Employés", "Retraités", "Cotisations", "Pensions", "Réserve"]
                table_data = []
                
                for year_data in avg_data:
                    table_data.append([
                        int(year_data['Year']),
                        int(year_data['TotEmp']),
                        int(year_data['TotRet']),
                        year_data['TotCotis'],
                        year_data['TotPens'],
                        year_data['Reserve']
                    ])
                
                pdf.add_table(headers, table_data, "Moyennes annuelles:")
        
        # Conclusions
        pdf.add_page()
        pdf.chapter_title("Conclusions et recommandations")
        
        conclusions_text = """Analyse comparative des scénarios:

1. Impact de l'âge de retraite:
   Le passage de 63 à 65 ans améliore significativement la situation financière du système.

2. Effet des cotisations:
   L'augmentation des taux de cotisation renforce la soutenabilité du système.

3. Impact des pensions:
   La réduction du coefficient de pension permet d'équilibrer partiellement les comptes.

4. Recommandations:
   - Envisager un relèvement progressif de l'âge de retraite
   - Ajuster les taux de cotisation selon la capacité contributive
   - Optimiser le niveau des pensions pour assurer la pérennité du système

Ce rapport fournit une base d'analyse pour les décisions de politique publique concernant le système de retraite.
"""
        pdf.chapter_body(conclusions_text)
        
        # Sauvegarder le PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"rapport_simulation_{timestamp}.pdf"
        pdf_path = os.path.join("static", pdf_filename)
        
        # Créer le dossier static s'il n'existe pas
        os.makedirs("static", exist_ok=True)
        
        pdf.output(pdf_path)
        return pdf_filename
        
    except Exception as e:
        print(f"Erreur lors de la génération du PDF: {e}")
        return None

# --- Version séquentielle optimisée ---
def run_simulation_with_germs_optimized(ix, iy, iz, selected_scenarios):
    """Version optimisée de la simulation complète"""
    
    all_scenarios_results = {}
    selected_scenario_ids = [int(scenario_id) for scenario_id in selected_scenarios]
    
    for scenario_id in selected_scenario_ids:
        print(f"Simulation du scénario {scenario_id}...")
        
        all_results = []
        current_ix, current_iy, current_iz = ix, iy, iz
        
        for i in range(NUM_SIMULATIONS):
            # Mettre à jour les graines
            current_ix = (current_ix + 5) % 30000 + 1
            current_iy = (current_iy + 5) % 30000 + 1
            current_iz = (current_iz + 5) % 30000 + 1
            
            result = run_single_scenario_simulation_optimized(scenario_id, (current_ix, current_iy, current_iz))
            all_results.append(pd.DataFrame(result))
            
            if (i + 1) % 10 == 0:
                print(f"  Progrès: {i + 1}/{NUM_SIMULATIONS}")
        
        all_scenarios_results[scenario_id] = all_results
    
    # Calculer les moyennes et intervalles de confiance
    averaged_results_by_scenario = {}
    confidence_intervals = {}
    Z_ALPHA_2 = 1.96
    
    for scenario_id, list_of_dfs in all_scenarios_results.items():
        # Moyennes
        concatenated_df = pd.concat(list_of_dfs)
        averaged_df = concatenated_df.groupby("Year").mean().reset_index()
        averaged_results_by_scenario[scenario_id] = averaged_df
        
        # Intervalles de confiance
        confidence_intervals[scenario_id] = {}
        for year_focus in [2025, 2030, 2035]:
            reserve_values = [df_run[df_run["Year"] == year_focus]["Reserve"].iloc[0] for df_run in list_of_dfs]
            mean_reserve = np.mean(reserve_values)
            std_dev = np.std(reserve_values)
            margin_of_error = Z_ALPHA_2 * (std_dev / math.sqrt(len(reserve_values)))
            confidence_intervals[scenario_id][year_focus] = {
                "Moyenne": mean_reserve,
                "IC_inf": mean_reserve - margin_of_error,
                "IC_sup": mean_reserve + margin_of_error,
                "Ecart-type": std_dev
            }
    
    # Générer les graphiques
    image_dir = "static/images"
    os.makedirs(image_dir, exist_ok=True)
    
    # 1. Graphique d'évolution des réserves
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (scenario_id, avg_df) in enumerate(averaged_results_by_scenario.items()):
        plt.plot(avg_df["Year"], avg_df["Reserve"] / 1000000000, 
                 label=SCENARIOS_PARAMS[scenario_id][0], 
                 marker='o', linestyle='-', 
                 color=colors[idx % len(colors)])
    
    plt.title("Évolution de la Réserve (Milliards de DH)", fontsize=14)
    plt.xlabel("Année")
    plt.ylabel("Réserve (Milliards de DH)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1))  
    plt.tight_layout()
    
    reserve_plot_filename = "reserve_evolution.png"
    reserve_plot_path = os.path.join(image_dir, reserve_plot_filename)
    plt.savefig(reserve_plot_path, dpi=100)
    plt.close()
    
    # 2. Tableaux récapitulatifs
    summary_tables = {}
    for year in [2025, 2030, 2035]:
        year_data = {}
        
        for scenario_id in selected_scenario_ids:
            scenario_name = SCENARIOS_PARAMS[scenario_id][0]
            scenario_results = []
            
            for sim_idx, df_run in enumerate(all_scenarios_results[scenario_id]):
                year_row = df_run[df_run["Year"] == year].iloc[0]
                scenario_results.append({
                    "Simulation": sim_idx + 1,
                    "TotEmp": year_row["TotEmp"],
                    "TotRet": year_row["TotRet"],
                    "TotCotis": year_row["TotCotis"],
                    "TotPens": year_row["TotPens"],
                    "Reserve": year_row["Reserve"],
                    "NouvRet": year_row["NouvRet"],
                    "NouvRec": year_row["NouvRec"]
                })
            
            df_scenario = pd.DataFrame(scenario_results)
            means = df_scenario.mean().to_dict()
            means["Simulation"] = "Moyenne"
            
            year_data[scenario_name] = {
                "simulations": scenario_results,
                "mean": means
            }
        
        summary_tables[year] = year_data
    
    # 3. Tableau des réserves par simulation
    reserve_tables = {}
    for scenario_id in selected_scenario_ids:
        scenario_name = SCENARIOS_PARAMS[scenario_id][0]
        reserve_data = []
        
        for sim_idx, df_run in enumerate(all_scenarios_results[scenario_id]):
            reserves = df_run[["Year", "Reserve"]].copy()
            reserves["Simulation"] = sim_idx + 1
            reserve_data.append(reserves)
        
        reserve_tables[scenario_name] = pd.concat(reserve_data)
    
    # 4. Graphiques comparatifs
    indicator_plots = {}
    indicators = ["TotEmp", "TotRet", "TotCotis", "TotPens", "Reserve", "NouvRet", "NouvRec"]
    
    for indicator in indicators:
        plt.figure(figsize=(12, 7))
        
        for idx, (scenario_id, avg_df) in enumerate(averaged_results_by_scenario.items()):
            plt.plot(avg_df["Year"], avg_df[indicator], 
                     label=SCENARIOS_PARAMS[scenario_id][0], 
                     marker='o', linestyle='-', 
                     color=colors[idx % len(colors)])
        
        plt.title(f"Évolution de {indicator}", fontsize=14)
        plt.xlabel("Année")
        plt.ylabel(indicator)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1))
        plt.tight_layout()
        
        filename = f"{indicator}_comparison.png"
        plot_path = os.path.join(image_dir, filename)
        plt.savefig(plot_path, dpi=100)
        plt.close()
        indicator_plots[indicator] = filename
    
    # 5. Tableau récapitulatif des réserves moyennes
    overall_reserve_table = []
    for year in range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1):
        year_data = {"Year": year}
        
        for scenario_id in selected_scenario_ids:
            scenario_name = SCENARIOS_PARAMS[scenario_id][0]
            avg_value = averaged_results_by_scenario[scenario_id][
                averaged_results_by_scenario[scenario_id]["Year"] == year
            ]["Reserve"].values[0]
            year_data[scenario_name] = avg_value
        
        overall_reserve_table.append(year_data)
    
    # 6. Tableau des intervalles de confiance
    confidence_table = []
    for year in [2025, 2030, 2035]:
        year_data = {"Year": year}
        
        for scenario_id in selected_scenario_ids:
            scenario_name = SCENARIOS_PARAMS[scenario_id][0]
            ci_data = confidence_intervals[scenario_id][year]
            year_data[scenario_name] = (
                f"[{ci_data['IC_inf']:.2f}, {ci_data['IC_sup']:.2f}] - "
                f"Moyenne: {ci_data['Moyenne']:.2f}"
            )
        
        confidence_table.append(year_data)
    
    # 7. Commentaires
    comments = "Analyse préliminaire :\n\n"
    comments += "- Scénario 2 (retraite à 65 ans) montre de meilleures réserves que le scénario 1\n"
    comments += "- L'augmentation des cotisations (Scénario 3) améliore significativement la santé financière\n"
    comments += "- La réduction des pensions (Scénario 4) compense partiellement les gains\n"
    
    # Préparer les résultats pour le PDF
    results_data = {}
    for scenario_id, avg_df in averaged_results_by_scenario.items():
        scenario_name = SCENARIOS_PARAMS[scenario_id][0]
        results_data[scenario_id] = {
            "name": scenario_name,
            "averages": avg_df.to_dict(orient='records'),
            "confidence": confidence_intervals.get(scenario_id, {})
        }

    # Générer le rapport PDF complet
    pdf_filename = generate_comprehensive_pdf_report(results_data, confidence_intervals, overall_reserve_table)

    return {
        "results": results_data,
        "summary_tables": summary_tables,
        "reserve_tables": {k: v.to_dict(orient='records') for k, v in reserve_tables.items()},
        "overall_reserve_table": overall_reserve_table,
        "confidence_table": confidence_table,
        "confidence_intervals": confidence_intervals,
        "comments": comments,
        "pdf_report": pdf_filename,
        "plots": {
            "reserve_evolution": reserve_plot_filename,
            "indicator_comparisons": indicator_plots    
        }
    }

# --- Application Flask ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        ix = int(request.form['germe_ix'])
        iy = int(request.form['germe_iy'])
        iz = int(request.form['germe_iz'])

        selected_scenarios = request.form.getlist('scenarios')

        if len(selected_scenarios) == 1 and ',' in selected_scenarios[0]:
            selected_scenarios = selected_scenarios[0].split(',')

        selected_scenario_ids = [int(scenario_id.strip()) for scenario_id in selected_scenarios]

        print(f"Démarrage de la simulation...")
        results = run_simulation_with_germs_optimized(ix, iy, iz, selected_scenario_ids)
        print("Simulation terminée!")

        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as e:
        print(f"Erreur: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    """Route pour télécharger le rapport PDF"""
    try:
        static_dir = os.path.join(os.getcwd(), 'static')
        file_path = os.path.join(static_dir, filename)
        
        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name=f"rapport_simulation_{datetime.now().strftime('%Y%m%d')}.pdf",
                mimetype='application/pdf'
            )
        else:
            return jsonify({"error": "Fichier PDF non trouvé"}), 404
            
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return jsonify({"error": f"Erreur lors du téléchargement: {str(e)}"}), 500

@app.route('/static/<filename>')
def serve_static_file(filename):
    """Route pour servir les fichiers statiques"""
    return send_from_directory('static', filename)

@app.route('/check_pdf/<filename>')
def check_pdf_exists(filename):
    """Route pour vérifier si le PDF existe"""
    try:
        static_dir = os.path.join(os.getcwd(), 'static')
        file_path = os.path.join(static_dir, filename)
        
        if os.path.exists(file_path):
            return jsonify({"exists": True, "size": os.path.getsize(file_path)})
        else:
            return jsonify({"exists": False})
            
    except Exception as e:
        return jsonify({"exists": False, "error": str(e)})

if __name__ == '__main__':
    # Créer les dossiers nécessaires
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    app.run(port=5000, debug=False)