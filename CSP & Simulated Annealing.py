"Scheduling AI: CSP & Simulated Annealing Stony Brook University Computer Engineering"

import random
import math
import time

backtrack_attempt = 0
sa_attempt = 0

# Define available timeslots
timeslots = list(range(10))

# Define 11 courses that need scheduling
courses = ['Class1', 'Class2', 'Class3', 'Class4',
           'Class5', 'Class6', 'Class7', 'Class8',
           'Class9', 'Class10', 'Class11']

# Mapping of each course to its corresponding professor
course_prof = {
    'Class1': 'ProfA',
    'Class2': 'ProfA',
    'Class3': 'ProfA',
    'Class4': 'ProfA',
    'Class5': 'ProfB',
    'Class6': 'ProfB',
    'Class7': 'ProfB',
    'Class8': 'ProfB',
    'Class9': 'ProfC',
    'Class10': 'ProfC',
    'Class11': 'ProfC'
}

# Available timeslots for each professor
prof_availability = {
    'ProfA': [0, 1, 2, 3, 4],
    'ProfB': [2, 3, 4, 5, 6],
    'ProfC': [5, 6, 7, 8, 9]
}

# Student course enrollments indicating which courses each student is taking.
# This setup is designed to be solvable.
student_enrollments = {
    'Student1': ['Class1', 'Class5', 'Class9'],
    'Student2': ['Class2', 'Class6', 'Class10'],
    'Student3': ['Class3', 'Class7', 'Class11'],
    'Student4': ['Class4', 'Class8'],
    'Student5': ['Class1', 'Class2', 'Class3']
}





def csp_schedule(courses, timeslots, course_prof, prof_availability, student_enrollments):
    assignment = {}  # # Dictionary to store course-to-timeslot assignments #Class 1: 3 , Class 2: 2......
    def is_valid(course, timeslot, assignment):
        # Check professor availability for the course.
        prof = course_prof[course]
        if timeslot not in prof_availability[prof]:
            return False

            # Check for professor conflict: if the professor teaches another course already scheduled at the same timeslot, then it is invalid.
        for other_course, other_timeslot in assignment.items():
            # If the same professor is teaching both courses, they must not be scheduled at the same time.
            if course_prof[other_course] == prof and timeslot == other_timeslot:
                return False

        # Check for conflicts among students enrolled in multiple courses
        for other_course, other_timeslot in assignment.items():
            for student, enrolled in student_enrollments.items():
                if course in enrolled and other_course in enrolled:
                    if timeslot == other_timeslot: # student has different classes with same time
                        return False
        return True

    def backtrack(assignment, courses_left):
        global backtrack_attempt
        # If there are no courses left to assign, return the current assignment as a complete solution.
        if not courses_left:
            return assignment
        # Select the next course to schedule.
        course = courses_left[0]
        # Try each timeslot for the current course.
        for timeslot in timeslots:
            backtrack_attempt += 1
            if is_valid(course, timeslot, assignment):
                #assign this timeslot to the course
                assignment[course] = timeslot
                # Recursively attempt to schedule the remaining courses.
                result = backtrack(assignment, courses_left[1:])
                if result is not None: #all courses have successfully been assigned
                    return result
                # Backtrack: Remove the assignment for the current course and try the next timeslot.
                del assignment[course]
        return None
    return backtrack(assignment, courses)



def simulated_annealing_schedule(courses, timeslots, course_prof, prof_availability, student_enrollments,
                                 initial_temp=1000, cooling_rate=0.9, min_temp=0, max_iter=1000):
    global sa_attempt

    # Initial solution: randomly assign a timeslot to each course
    current_solution = {course: random.choice(timeslots) for course in courses}

    def cost(solution):
        total_cost = 0
        # Check professor availability violations: add penalty if a course is scheduled outside the professor's available timeslots # penalty 5 for professor availability violation
        for course, timeslot in solution.items():
            prof = course_prof[course]
            if timeslot not in prof_availability[prof]:
                total_cost += 5

        # Check for professor conflict: if two courses taught by the same professor are scheduled in the same timeslot
        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                course1 = courses[i]
                course2 = courses[j]
                # If the courses are taught by the same professor and scheduled at the same timeslot, add a penalty.
                if course_prof[course1] == course_prof[course2] and solution[course1] == solution[course2]:
                    total_cost += 5  # Penalty for professor conflict

        # Check for student conflicts: for each student, add a penalty for every pair of courses scheduled at the same timeslot #penalty 1 for student class conflict
        for student, enrolled in student_enrollments.items():
            n = len(enrolled)
            for i in range(n):
                for j in range(i + 1, n):
                    c1 = enrolled[i]
                    c2 = enrolled[j]
                    if solution[c1] == solution[c2]:
                        total_cost += 1
        return total_cost

    # Evaluate the cost of the initial solution
    current_cost = cost(current_solution)
    T = initial_temp
    iter_count = 0

    best_solution = current_solution.copy()
    best_cost = current_cost

    # Main loop of the simulated annealing algorithm
    while T > min_temp and iter_count < max_iter:
        # Generate a neighbor solution by randomly selecting a course and changing its timeslot
        neighbor = current_solution.copy()
        course_to_change = random.choice(courses)
        new_timeslot = random.choice(timeslots)
        neighbor[course_to_change] = new_timeslot

        neighbor_cost = cost(neighbor)
        delta = neighbor_cost - current_cost

        # Accept the neighbor if it is better, or with a probability that decreases with worse cost and lower temperature
        if delta < 0 or random.random() < math.exp(-delta /(1.38e-23) * T):
            current_solution = neighbor
            current_cost = neighbor_cost

            # Update the best solution if the current solution improves the best cost found so far
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
            sa_attempt += 1
            if best_cost == 0:
                return best_solution, best_cost

        # Cool down the temperature
        T *= cooling_rate
        iter_count += 1

    return best_solution, best_cost




if __name__ == "__main__":
    print("【CSP Scheduling Solution】")
    csp_solution = csp_schedule(courses, timeslots, course_prof, prof_availability, student_enrollments)
    print("Final Assignment:", csp_solution)
    print(f"assignment attempted:{backtrack_attempt} times")

    print("\n【Simulated Annealing Scheduling Solution】")
    sa_solution, sa_cost = simulated_annealing_schedule(courses, timeslots, course_prof, prof_availability, student_enrollments)
    print("Final Assignment:", sa_solution)
    print("cost:", sa_cost)
    print(f"assignment attempted:{sa_attempt} times")
