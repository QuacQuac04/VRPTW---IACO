import random
import numpy as np
import matplotlib.pyplot as plt

class Customer:
    def __init__(self, id, x, y, demand, ready_time, due_date, service_time):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

class Vehicle:
    def __init__(self, capacity):
        self.capacity = capacity
        self.route = []
        self.current_load = 0
        self.current_time = 0

    def __str__(self):
        return "Route: " + " -> ".join([str(customer.id) if not isinstance(customer, Depot) else "Depot" for customer in self.route])

class VRPTW:
    def __init__(self, customers, depot, max_iterations, max_fe, num_ants, alpha, beta, rho, num1):
        self.customers = customers
        self.depot = depot
        self.max_iterations = max_iterations
        self.max_fe = max_fe
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num1 = num1
        self.pheromone = np.ones((len(customers), len(customers)))  # Pheromone matrix
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_vehicle_count = float('inf')

    def distance(self, customer1, customer2):
        return np.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)

    def evaluate_solution(self, solution):
        total_distance = 0
        total_vehicles = 0
        for vehicle in solution:
            total_distance += self.calculate_route_distance(vehicle.route)
            if vehicle.route:  # Count only if the vehicle has a route
                total_vehicles += 1
        return total_distance, total_vehicles

    def calculate_route_distance(self, route):
        distance = 0
        current_time = 0
        for i in range(len(route)):
            if isinstance(route[i], Depot):  # Kiểm tra nếu là Depot
                continue  # Bỏ qua Depot trong tính toán
            if i == 0:
                distance += self.distance(self.depot, route[i])
            else:
                distance += self.distance(route[i-1], route[i])
            current_time += route[i].service_time
        distance += self.distance(route[-1], self.depot)  # Return to depot
        return distance

    def update_pheromone(self, solutions):
        for solution in solutions:
            distance, vehicles = self.evaluate_solution(solution)
            for vehicle in solution:
                for i in range(len(vehicle.route) - 1):
                    # Kiểm tra chỉ số hợp lệ trước khi cập nhật pheromone
                    if isinstance(vehicle.route[i], Depot) or isinstance(vehicle.route[i + 1], Depot):
                        continue  # Bỏ qua nếu là Depot
                    if vehicle.route[i].id < len(self.pheromone) and vehicle.route[i + 1].id < len(self.pheromone):
                        self.pheromone[vehicle.route[i].id][vehicle.route[i + 1].id] += 1 / distance

    def run(self):
        for iteration in range(self.max_iterations):
            solutions = []
            for ant in range(self.num_ants):
                solution = self.construct_solution()
                solutions.append(solution)
                distance, vehicles = self.evaluate_solution(solution)
                if (vehicles < self.best_vehicle_count) or (vehicles == self.best_vehicle_count and distance < self.best_cost):
                    self.best_cost = distance
                    self.best_vehicle_count = vehicles
                    self.best_solution = solution
            self.update_pheromone(solutions)
            self.pheromone *= self.rho  # Evaporation

    def construct_solution(self):
        vehicles = [Vehicle(capacity=4000) for _ in range(self.num1)]  # Tạo danh sách xe
        unvisited_customers = self.customers.copy()  # Danh sách khách hàng chưa được phục vụ
        for vehicle in vehicles:
            current_location = self.depot  # Bắt đầu từ kho
            while unvisited_customers:
                # Tìm khách hàng gần nhất có thể phục vụ
                next_customer = None
                min_distance = float('inf')
                for customer in unvisited_customers:
                    distance = self.distance(current_location, customer)
                    if distance < min_distance and vehicle.current_load + customer.demand <= vehicle.capacity:
                        min_distance = distance
                        next_customer = customer
                
                if next_customer is None:  # Không còn khách hàng nào có thể phục vụ
                    break
                
                # Cập nhật lộ trình của xe
                vehicle.route.append(next_customer)
                vehicle.current_load += next_customer.demand
                vehicle.current_time += self.distance(current_location, next_customer) + next_customer.service_time
                current_location = next_customer  # Cập nhật vị trí hiện tại
                unvisited_customers.remove(next_customer)  # Xóa khách hàng đã phục vụ

            # Trở về kho
            vehicle.route.append(self.depot)
        
        return vehicles  # Trả về danh sách các xe với lộ trình của chúng

class Depot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def read_solomon_data(file_path):
    customers = []
    depot = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Bỏ qua các dòng không chứa dữ liệu
            if not line.strip() or line.startswith('C') or line.startswith('VEHICLE') or line.startswith('CUSTOMER'):
                continue
            
            data = line.split()
            # Kiểm tra xem có đủ dữ liệu để tạo depot
            if depot is None and len(data) >= 3:
                depot = Depot(float(data[1]), float(data[2]))  # Tọa độ kho
            elif len(data) >= 7:  # Kiểm tra số lượng cột cho khách hàng
                customers.append(Customer(
                    id=int(data[0]),
                    x=float(data[1]),
                    y=float(data[2]),
                    demand=float(data[3]),
                    ready_time=float(data[4]),
                    due_date=float(data[5]),
                    service_time=float(data[6])
                ))
    return customers, depot

def calculate_distance(customer1, customer2):
    return np.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)

# Đọc dữ liệu từ tệp Solomon
customers, depot = read_solomon_data('path_to_solomon_data3.txt')

# Ví dụ tính toán khoảng cách giữa kho và khách hàng đầu tiên
distance_to_first_customer = calculate_distance(depot, customers[0])
print(f"Distance from depot to customer 0: {distance_to_first_customer:.2f}")

# Cài đặt tham số
max_iterations = 100
max_fe = 300000
num_ants = 50
alpha = 1
beta = 2
rho = 0.9
num1 = 20  # Số lượng thành phố ưu tiên trong khởi tạo

# Khởi chạy thuật toán
vrptw = VRPTW(customers, depot, max_iterations, max_fe, num_ants, alpha, beta, rho, num1)
vrptw.run()

# In kết quả
print("Chi phi:", vrptw.best_cost)
print("So luong xe tot nhat:", vrptw.best_vehicle_count)
print("Duong di toi uu:")
for vehicle in vrptw.best_solution:
    print(vehicle)

# Vẽ lộ trình
plt.figure(figsize=(10, 6))
for vehicle in vrptw.best_solution:
    route_x = [depot.x] + [customer.x for customer in vehicle.route if not isinstance(customer, Depot)] + [depot.x]
    route_y = [depot.y] + [customer.y for customer in vehicle.route if not isinstance(customer, Depot)] + [depot.y]
    plt.plot(route_x, route_y, marker='o')
    for customer in vehicle.route:
        if not isinstance(customer, Depot):
            plt.text(customer.x, customer.y, str(customer.id), fontsize=12, ha='right')

plt.title('Tuyến Đường Đi Tốt Nhất')
plt.xlabel('Trục X')
plt.ylabel('Trục Y')
plt.grid()
plt.show()