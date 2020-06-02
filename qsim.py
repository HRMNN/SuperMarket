# DataFrames
import pandas as pd
# Visualizations
import matplotlib.pyplot as plt
# Statistics
from scipy import stats 
# Array and Array Math
import numpy as np

# Random generates an Entry-Process (= "the time a customer enters the waiting area")
def entry_process(customers, entry_variable, entry_variable_parameters):
    # Set up Collector for results
    results          = pd.DataFrame([])
    # Generates randomly how much time passes between Customers
    results["time"]  = entry_variable(** entry_variable_parameters, size = customers)
    # Sum up to determine entry time of each customer
    results["entry"] = results.time.cumsum()
    # Return results
    return results["entry"]

# Random-generates the the service-process and the resulting exit-process
# (= "the time a customer gets to/ leaves the cashier")
def service_process(entries, service_variable, service_variable_parameters, shifts):
    # Turn Shifts into Stations
    stations = np.array(shifts, dtype=[('ready', '<f8'), ('close', '<f8')])
    # Set up Collectors for Results
    results_service = ([])
    results_station = ([])
    results_exit    = ([]) 
    # Cycle through Customers of the Entry-Process
    for idx in entries:      
        # Index of next available Station
        # (Customer chooses Station which has been unoccupied the longest)
        free = np.nanmin(np.argwhere(stations["ready"] == np.nanmin(stations["ready"])))
        # Add Time when Customer starts being serviced 
        results_service.append(max(idx,stations["ready"][free])) 
        # Generate time when Customer is done
        done = float(max(idx,stations["ready"][free]) + service_variable(** service_variable_parameters, size = 1))
        # Add Time when Customer is done/ leaves the register
        results_exit.append(done)
        # Add used Station
        results_station.append(free)
        # Check if shift is over and Station can be closed
        if(stations["close"][free] <= done):
            # Close Station
            stations["ready"][free] = np.nan
        else:
            # Update Station's Ready Time
            stations["ready"][free] = done   
    # Return results
    return results_service, results_exit, results_station

# Simulate Queue by combining entry_process and service_process, and returning the relevant data in a dataframe
def makeq(n_customers,
          entry_variable,entry_variable_parameters,
          service_variable,service_variable_parameters,
          shifts,
          exitsort = True):
    # Set Up Collector for Results
    overview                              = pd.DataFrame([])
    # Random Generate Entrance of Customers
    overview["entry"]                     = entry_process(n_customers,
                                                          entry_variable,
                                                          entry_variable_parameters)
    # Random Generate Service/ Exit of Customers
    overview["service"], overview["exit"], overview["station"] = service_process(overview["entry"], 
                                                                                 service_variable,
                                                                                 service_variable_parameters, 
                                                                                 shifts)
    # If chosen, sort by Exit-Time
    if exitsort:
        overview.sort_values(by=["exit"], inplace = True)
        overview.reset_index(inplace = True)        
    # Return resulting Dataframe
    return(overview)

# Make a set, which stores the relevant results (exit time of each costumer) from multiple queuing simulation runs
def alotta_queues (runs,
                 n_customers, entry_variable, entry_variable_parameters,
                 service_variable, service_variable_parameters, shifts):
    # Set up Collector for results
    exits = pd.DataFrame([])
    # Runs multiple Queing Simulations
    for idx in range(1,runs+1):    
        exits = exits.append(makeq(n_customers, entry_variable, entry_variable_parameters,
                                  service_variable, service_variable_parameters, shifts).exit)
    exits.index = range(0,runs)
    exits = exits.T
    return exits


# Visualize the Runs of the Queuing Simulations
def show_mcq (qspace, title, breakoff):
    # Extract necessary Information
    runs        = qspace.shape[1]
    n_customers = qspace.shape[0]   
    # Process Data for Visualization
    qspace_max = qspace.max(axis = 1)
    qspace_q99 = qspace.quantile(.99, axis = 1)
    qspace_q75 = qspace.quantile(.75, axis = 1)
    qspace_med = qspace.median(axis = 1)
    qspace_q25 = qspace.quantile(.25, axis = 1)
    qspace_q01 = qspace.quantile(.01, axis = 1)
    qspace_min = qspace.min(axis = 1)
    # Visualize Results
    plt.suptitle(title)
    plt.title(str(runs)+" runs")
    plt.fill_between(x = range(0,n_customers), y1 = qspace_max, y2 = qspace_min, color = "#DDEEDD")
    plt.fill_between(x = range(0,n_customers), y1 = qspace_q99, y2 = qspace_q01, color = "#BBCCBB")
    plt.fill_between(x = range(0,n_customers), y1 = qspace_q75, y2 = qspace_q25, color = "#99AA99")
    plt.plot(qspace_med, color = "green")
    plt.ylabel("Time of Exit")
    plt.xlabel("Customer")
    plt.legend(['Median',"Minimum to Maximum",'Quantile .01 to .99','Quantile .25 to .75'], loc = 2)
    # If chosen, the graph can be closed of for additions
    if breakoff:
        plt.show()
        
# Compare a certain case to model-based generated data
def compare(qspace, case, visualize, title):
    # Visualize if Chosen
    if visualize:
        # Visualize Queing System
        show_mcq(qspace, title, False)
        # Visualize given Case
        plt.plot(case.exit, color = 'red')
        plt.show()
        # Show Case
        plt.title("Z-Score")
        plt.ylabel("Z-Score")
        plt.xlabel("Customer")        
        plt.plot((case.exit - qspace.mean(axis = 1)) / qspace.std(axis = 1))     
        # Visualize Mean
        plt.hlines(((case.exit - qspace.mean(axis = 1)) / qspace.std(axis = 1)).mean(), xmin = 0, xmax = case.shape[0], color = "orange")
        plt.legend(['Z-Score of each Customer','Average of all customers Z-Scores'])
        plt.show()
    # Return Mean Square Error
    return ((case.exit - qspace.mean(axis = 1)) / qspace.std(axis = 1)).mean()