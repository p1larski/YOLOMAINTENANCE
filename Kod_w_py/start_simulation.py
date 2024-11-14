import csv
import random
import numpy as np
import os

# Zmienna cycles na początku programu
cycles = 5000  # Liczba cykli do wygenerowania

# Funkcje symulujące odchylenia w pracy wtryskarki
def simulate_injection_molding_cycle(cycle_id):
    welding_time = np.random.normal(5, 0.5)  # Czas zgrzewania (sekundy)
    pressure = np.random.normal(100, 10)  # Ciśnienie (jednostki umowne)
    energy = np.random.normal(50, 5)  # Zużyta energia (jednostki umowne)
    temperature = np.random.normal(200, 10)  # Temperatura (°C)
    ultrasonic_power = np.random.normal(90, 5)  # Moc ultradźwięków (jednostki umowne)
    humidity = np.random.normal(45, 5)  # Wilgotność otoczenia (%)
    vibration = np.random.normal(0.02, 0.005)  # Wibracje maszyny (jednostki umowne)
    cycle_time = np.random.normal(12, 1)  # Czas pełnego cyklu (sekundy)
    operating_pressure = np.random.normal(120, 8)  # Ciśnienie robocze (jednostki umowne)
    
    # Dodatkowe parametry wtryskarki
    form_closing_time = np.random.normal(2.0, 0.2)  # Czas zamknięcia formy (sekundy)
    form_closing_force = np.random.normal(500, 20)  # Siła zamknięcia formy (jednostki umowne)
    material_flow_rate = np.random.normal(10, 1)  # Przepływ materiału (jednostki umowne)
    tool_wear = np.random.normal(0.5, 0.1)  # Zużycie narzędzi (procenty)
    maintenance_status = np.random.choice(["1", "0"], p=[0.95, 0.05])
    power_consumption = np.random.normal(100, 10)  # Pobór mocy (kW)
    voltage_fluctuation = np.random.normal(230, 5)  # Wahania napięcia (V)
    
    flatness = 'flat' if welding_time < 6 and temperature < 210 else 'not flat'
    
    if random.random() < 0.05:  # 5% szansa na problem techniczny
        pressure *= 0.8
        flatness = 'not flat'
    
    return {
        "Cycle ID": cycle_id,
        "Welding Time": welding_time,
        "Pressure": pressure,
        "Energy": energy,
        "Temperature": temperature,
        "Ultrasonic Power": ultrasonic_power,
        "Humidity": humidity,
        "Vibration": vibration,
        "Cycle Time": cycle_time,
        "Operating Pressure": operating_pressure,
        "Flatness": flatness,
        "Form Closing Time": form_closing_time,
        "Form Closing Force": form_closing_force,
        "Material Flow Rate": material_flow_rate,
        "Tool Wear": tool_wear,
        "Maintenance Status": maintenance_status,
        "Power Consumption": power_consumption,
        "Voltage Fluctuation": voltage_fluctuation
    }

# Funkcje symulujące pracę robota
def simulate_robot_operation(cycle_id, flatness):
    suction_power = np.random.normal(80, 5)
    positioning_error = np.random.normal(0, 0.02)
    vacuum_pressure = np.random.normal(95, 5)
    speed = np.random.normal(2.5, 0.2)
    tilt_angle = np.random.normal(0, 0.5)
    suction_force = np.random.normal(85, 4)
    operation_time = np.random.normal(3.5, 0.3)
    robot_vibration = np.random.normal(0.015, 0.003)
    robot_temperature = np.random.normal(25, 2)
    
    suction_system_health = np.random.choice(["1", "0"], p=[0.97, 0.03])
    robot_battery_level = np.random.normal(90, 5)
    robot_arm_torque = np.random.normal(50, 5)
    robot_joint_wear = np.random.normal(0.1, 0.02)
    
    will_fall = False
    if flatness == 'not flat' or suction_power < 75 or positioning_error > 0.03 or tilt_angle > 5:
        will_fall = True
    
    if random.random() < 0.02:
        will_fall = True
    
    return {
        "Cycle ID": cycle_id,
        "Suction Power": suction_power,
        "Positioning Error": positioning_error,
        "Vacuum Pressure": vacuum_pressure,
        "Speed": speed,
        "Tilt Angle": tilt_angle,
        "Suction Force": suction_force,
        "Operation Time": operation_time,
        "Robot Vibration": robot_vibration,
        "Robot Temperature": robot_temperature,
        "Element Dropped": will_fall,
        "Suction System Health": suction_system_health,
        "Robot Battery Level": robot_battery_level,
        "Robot Arm Torque": robot_arm_torque,
        "Robot Joint Wear": robot_joint_wear
    }

# Funkcja do symulacji cykli produkcyjnych i zapisu do dwóch oddzielnych plików CSV
def simulate_production_cycles(num_cycles, output_injection_file, output_robot_file):
    # Tworzenie katalogu, jeśli nie istnieje
    os.makedirs(os.path.dirname(output_injection_file), exist_ok=True)
    
    with open(output_injection_file, mode='w', newline='') as injection_file, \
         open(output_robot_file, mode='w', newline='') as robot_file:
        
        injection_writer = csv.DictWriter(injection_file, fieldnames=[
            "Cycle ID", "Welding Time", "Pressure", "Energy", "Temperature", "Ultrasonic Power", 
            "Humidity", "Vibration", "Cycle Time", "Operating Pressure", "Flatness",
            "Form Closing Time", "Form Closing Force", "Material Flow Rate", "Tool Wear", 
            "Maintenance Status", "Power Consumption", "Voltage Fluctuation"
        ])
        injection_writer.writeheader()

        robot_writer = csv.DictWriter(robot_file, fieldnames=[
            "Cycle ID", "Suction Power", "Positioning Error", "Vacuum Pressure", "Speed", 
            "Tilt Angle", "Suction Force", "Operation Time", "Robot Vibration", "Robot Temperature", 
            "Element Dropped", "Suction System Health", "Robot Battery Level", "Robot Arm Torque", 
            "Robot Joint Wear"
        ])
        robot_writer.writeheader()
        
        for cycle_id in range(1, num_cycles + 1):
            # Symulacja wtryskarki
            injection_data = simulate_injection_molding_cycle(cycle_id)
            injection_writer.writerow(injection_data)
            
            # Symulacja robota
            robot_data = simulate_robot_operation(cycle_id, injection_data['Flatness'])
            robot_writer.writerow(robot_data)

            # Wyświetlanie postępu
            progress = (cycle_id / num_cycles) * 100
            print(f"Progress: {progress:.2f}% - Cycle ID: {cycle_id}/{num_cycles}", end="\r")

    print(f"\nSymulacja zakończona. Dane zapisane do plików: {output_injection_file} oraz {output_robot_file}")

# Ustawienie ścieżki dla plików CSV
output_dir = "BRANSON/files"
injection_file_path = os.path.join(output_dir, "injection_traceability.csv")
robot_file_path = os.path.join(output_dir, "robot_traceability.csv")

# Uruchomienie symulacji
simulate_production_cycles(num_cycles=cycles, output_injection_file=injection_file_path, output_robot_file=robot_file_path)
