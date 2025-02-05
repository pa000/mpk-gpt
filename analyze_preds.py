# Made with ChatGPT's help

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def time_to_minutes(time_str):
    """Konwertuje czas w formacie %H:%M na minuty od północy."""
    return int(time_str.split(':')[0]) * 60 + int(time_str.split(':')[1])

def time_to_hour(time_str):
    """Konwertuje czas w formacie %H:%M na godzinę (0-23)."""
    return int(time_str.split(':')[0])

def calculate_errors(df):
    """Oblicza błędy predykcji w minutach oraz dodaje godziny do analizy."""
    df['real_minutes'] = df['real_time'].apply(time_to_minutes)
    df['pred_minutes'] = df['pred_time'].apply(time_to_minutes)
    df['error'] = df['pred_minutes'] - df['real_minutes']
    df['real_hour'] = df['real_time'].apply(time_to_hour)  # Dodajemy godzinę do analizy
    df['pred_hour'] = df['pred_time'].apply(time_to_hour)
    return df

def process_files():
    files = glob('preds_*.csv')
    all_step_accuracies = {}
    all_hour_accuracies = {}
    all_hour_accuracies_filtered = {}
    model_labels = []
    
    for file in files:
        model_name = file.replace('preds_', '').replace('.csv', '')
        df = pd.read_csv(file)
        df = calculate_errors(df)
        model_labels.append(model_name)
        
        # Analiza dokładności w zależności od liczby przystanków
        steps = sorted(df['steps'].unique())
        for step in steps:
            step_df = df[df['steps'] == step]
            exact_predictions = (step_df['error'] == 0).sum()
            
            if step not in all_step_accuracies:
                all_step_accuracies[step] = {}
            all_step_accuracies[step][model_name] = exact_predictions
        
        # Analiza dokładności w zależności od godziny (procentowo)
        hours = sorted(df['real_hour'].unique())
        for hour in hours:
            hour_df = df[df['real_hour'] == hour]
            total_predictions = len(hour_df)
            exact_predictions_hour = (hour_df['error'] == 0).sum()
            
            # Obliczamy dokładność w procentach
            accuracy_percentage = (exact_predictions_hour / total_predictions) * 100 if total_predictions > 0 else 0
            
            if hour not in all_hour_accuracies:
                all_hour_accuracies[hour] = {}
            all_hour_accuracies[hour][model_name] = accuracy_percentage

        # Analiza dokładności tylko dla przewidywań do 5 przystanków
        filtered_df = df[df['steps'] <= 5]  # Filtrujemy tylko przewidywania do 5 przystanków
        for hour in hours:
            hour_filtered_df = filtered_df[filtered_df['real_hour'] == hour]
            total_predictions_filtered = len(hour_filtered_df)
            exact_predictions_filtered = (hour_filtered_df['error'] == 0).sum()
            
            # Obliczamy dokładność w procentach dla danych filtrowanych
            accuracy_percentage_filtered = (exact_predictions_filtered / total_predictions_filtered) * 100 if total_predictions_filtered > 0 else 0
            
            if hour not in all_hour_accuracies_filtered:
                all_hour_accuracies_filtered[hour] = {}
            all_hour_accuracies_filtered[hour][model_name] = accuracy_percentage_filtered
    
    # Tworzenie zbiorczego wykresu dokładności w zależności od liczby przystanków
    plt.figure(figsize=(14, 8))  # Powiększenie rozmiaru wykresu
    for model_name in model_labels:
        step_values = [all_step_accuracies[step].get(model_name, 0) for step in sorted(all_step_accuracies.keys())]
        plt.plot(sorted(all_step_accuracies.keys()), step_values, marker='o', markersize=4, label=model_name)  # Zmniejszenie rozmiaru kropek
    
    plt.xlabel("Liczba przystanków do przodu")
    plt.ylabel("Liczba dokładnych przewidywań")
    plt.legend()
    plt.title("Zbiorczy wykres dokładności przewidywań w zależności od liczby przystanków")
    plt.savefig("accuracy_vs_steps_all_models.png", dpi=300)  # Większa rozdzielczość
    plt.show()

    # Tworzenie wykresu dokładności w zależności od godziny (procentowo)
    plt.figure(figsize=(14, 8))  # Powiększenie rozmiaru wykresu
    for model_name in model_labels:
        hour_values = [all_hour_accuracies[hour].get(model_name, 0) for hour in sorted(all_hour_accuracies.keys())]
        plt.plot(sorted(all_hour_accuracies.keys()), hour_values, marker='o', markersize=4, label=model_name)  # Zmniejszenie rozmiaru kropek
    
    plt.xlabel("Godzina dnia")
    plt.ylabel("Procent dokładnych przewidywań")
    plt.legend()
    plt.title("Zbiorczy wykres dokładności przewidywań w zależności od godziny (procentowo)")
    plt.savefig("accuracy_vs_hour_percentage_all_models.png", dpi=300)  # Większa rozdzielczość
    plt.show()

    # Tworzenie wykresu dokładności w zależności od godziny (procentowo, tylko do 5 przystanków)
    plt.figure(figsize=(14, 8))  # Powiększenie rozmiaru wykresu
    for model_name in model_labels:
        hour_values_filtered = [all_hour_accuracies_filtered[hour].get(model_name, 0) for hour in sorted(all_hour_accuracies_filtered.keys())]
        plt.plot(sorted(all_hour_accuracies_filtered.keys()), hour_values_filtered, marker='o', markersize=4, label=model_name)  # Zmniejszenie rozmiaru kropek
    
    plt.xlabel("Godzina dnia")
    plt.ylabel("Procent dokładnych przewidywań (do 5 przystanków)")
    plt.legend()
    plt.title("Zbiorczy wykres dokładności przewidywań w zależności od godziny (procentowo, do 5 przystanków)")
    plt.savefig("accuracy_vs_hour_percentage_filtered_all_models.png", dpi=300)  # Większa rozdzielczość
    plt.show()

if __name__ == "__main__":
    process_files()
