# helper functions for creating multilayer networks

# import libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import networkx as nx
import osmnx as ox

# calculate cosine from three points
def cos_between_points(A, B, C):
    '''
    Calculates the cosine of vectors AB and BC.
    If A, B, C are alingned on a straight line it will return 1.
    '''
    # Convert Shapely Points to numpy arrays
    A_np = np.array([A.x, A.y])
    B_np = np.array([B.x, B.y])
    C_np = np.array([C.x, C.y])
    
    # Compute vectors AB and BC
    AB = B_np - A_np
    BC = C_np - B_np
    
    # Calculate dot product of AB and BC
    dot_product = np.dot(AB, BC)
    
    # Calculate the magnitudes of vectors AB and BC
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)
    
    # Calculate the angle in radians using the dot product formula
    cos_theta = dot_product / (magnitude_AB * magnitude_BC)
    
    return cos_theta



# function that creates edge list from KSJ
def create_edge_list(routes, stations, tolerance = 10):
    '''
    Creates an edge list from routes and stations.

    Parameters
    ----------
    routes : geopandas.GeoDataFrame
        GeoDataFrame of routes as a subset of the KSJ data file.
    stations : geopandas.GeoDataFrame
        GeoDataFrame of stations as a subset of the KSJ data file.
    tolerance : int
        The distance in meters to allow between station and the line.

    Returns
    -------
    edge_list : geopandas.GeoDataFrame
        GeoDataFrame of edge list of all routes.

    '''

    # get list of operators
    operators_list = list(routes['N05_003'].unique())

    # initialise edge list
    edge_dfs = []

    # iterate through each operator
    for op in operators_list:
        print(f'Processing {op}')
        # extract routes and stations by operator
        routes_op = routes[routes['N05_003'] == op].copy()
        stations_op = stations[stations['N05_003'] == op].copy()

        # extract only the necessary columns
        station_columns = ['N05_002', 'N05_003', 'N05_006', 'N05_011', 'geometry']
        stations_op = stations_op[station_columns].copy()

        # get list of lines within operator
        lines_list = list(routes_op['N05_002'].unique())


        # iterate through each line
        for line in lines_list:
            print(f'Working on {line}')
            # extract routes and stations for each line
            routes_temp = routes_op[routes_op['N05_002'] == line].copy()
            stations_temp = stations_op[stations_op['N05_002'] == line].copy()

            # add column that shows these stations are real stations
            stations_temp['real_flag'] = True

            # check if there are stations in the endpoints of each of the geometry and add to dataframe if not
            point_list = []
            all_stations = stations_temp['geometry'].unary_union
            for idx, row in routes_temp.iterrows():
                # store endpoints
                endpoints = shapely.get_parts(row['geometry'].boundary)
                # check if endpoints are stations and if not, add to the point list (if not already added)
                for p in endpoints:
                    if not (p.dwithin(all_stations, tolerance) | p.dwithin(shapely.MultiPoint(point_list), tolerance)):
                        point_list.append(p)
            

            # create gdf of additional stations
            add_stations = gpd.GeoDataFrame({
                'N05_002': line,
                'N05_003': op,
                'geometry': point_list,
                'N05_006': 0
            }, crs = stations.crs).reset_index().rename(columns = {'index': 'N05_011'})
            # change name of imaginary stations
            add_stations['N05_011'] = 'station' + add_stations['N05_011'].astype(str)
            # merge with real stations
            stations_temp = gpd.GeoDataFrame(pd.concat([stations_temp, add_stations]))
            # print(stations_temp['N05_011'].to_list())

            # get list of imaginary stations
            st_list = add_stations['N05_011'].to_list()

            # now that we are sure all edges end with nodes, we can create the edge lists
            edge_list_dfs = []

            # iterate through the rows again
            for idx, row in routes_temp.iterrows():
                # get geometry
                line_geom = row['geometry']
                
                # extract points that are on the line
                st = stations_temp[stations_temp['geometry'].apply(lambda x: x.dwithin(line_geom, tolerance))].copy()
                # print(st['N05_011'].to_list())

                # add distance from starting point and sort in order
                st['distance'] = line_geom.project(st['geometry'])
                st.sort_values('distance', inplace = True)

                # create edge list
                edges_temp = pd.DataFrame.from_dict({
                    'source_name': st['N05_011'].to_list()[:-1],
                    'target_name': st['N05_011'].to_list()[1:],
                    'source_ID': st['N05_006'].to_list()[:-1],
                    'target_ID': st['N05_006'].to_list()[1:],
                    'source_dist': st['distance'].to_list()[:-1],
                    'target_dist': st['distance'].to_list()[1:],
                })

                # get distance and geometry
                edges_temp['distance'] = edges_temp['target_dist'] - edges_temp['source_dist']
                edges_temp['geometry'] = edges_temp.apply(lambda r: shapely.ops.substring(line_geom, r['source_dist'], r['target_dist']), axis = 1)
                
                # drop the source_dist and target_dist
                edges_temp.drop(columns = ['source_dist', 'target_dist'], inplace = True)

                # add to list
                edge_list_dfs.append(edges_temp)

            # create edge list
            edge_list = pd.concat(edge_list_dfs, ignore_index = True)

            # iterate through imaginary stations
            for st_temp in st_list:
                # select rows including the station
                edge_list_temp = edge_list[(edge_list['source_name'] == st_temp) | (edge_list['target_name'] == st_temp)].copy()
                # drop these columns from the original data frame
                edge_list = edge_list[(edge_list['source_name'] != st_temp) & (edge_list['target_name'] != st_temp)].copy()

                # flip so that all rows end with the imaginary station
                for idx, row in edge_list_temp.iterrows():
                    if row['source_name'] == st_temp:

                        # store the values
                        other = row['target_name']
                        other_ID = row['target_ID']
                        geometry = row['geometry'].reverse()

                        # replace values
                        edge_list_temp.loc[idx, 'source_name'] = other
                        edge_list_temp.loc[idx, 'target_name'] = st_temp
                        edge_list_temp.loc[idx, 'source_ID'] = other_ID
                        edge_list_temp.loc[idx, 'target_ID'] = 0
                        edge_list_temp.loc[idx, 'geometry'] = geometry

                # initialise dictionary that we will later turn into gdf
                new_edge_dict = {
                    'source_name': [],
                    'target_name': [],
                    'source_ID': [],
                    'target_ID': [],
                    'distance': [],
                    'geometry': []
                }

                # calculate the angle between each segment and merge geometry if smooth
                length = edge_list_temp.shape[0]
                for i in np.arange(length):
                    for j in np.arange(i + 1, length):
                        # station name and whether imaginary or real for each pair of nodes
                        st1 = edge_list_temp.iloc[i].loc['source_name']
                        st2 = edge_list_temp.iloc[j].loc['source_name']
                        st_id1 = edge_list_temp.iloc[i].loc['source_ID']
                        st_id2 = edge_list_temp.iloc[j].loc['source_ID']

                        # get geometry of the two things
                        geom1 = edge_list_temp.iloc[i].loc['geometry']
                        geom2 = edge_list_temp.iloc[j].loc['geometry']


                        # get end point
                        top_node = shapely.get_point(geom1, -1)
                        node_1 = shapely.get_point(geom1, -2)
                        node_2 = shapely.get_point(geom2, -2)

                        # calculate cosine
                        cos = cos_between_points(node_1, top_node, node_2)

                        # if smooth (larger than 0) connect the geometries
                        if cos > 0:
                            # shapely.line_merge does not work - the subtle mismatch does not allow for merging
                            # merged_geom = shapely.line_merge(shapely.MultiLineString([geom1, geom2.reverse()]))
                            points = [shapely.Point(xy) for xy in geom1.coords] + [shapely.Point(xy) for xy in geom2.reverse().coords]
                            merged_geom = shapely.LineString(points)

                            # calculate distance
                            merged_distance = edge_list_temp.iloc[i].loc['distance'] + edge_list_temp.iloc[j].loc['distance']
                            
                            # add to the dictionary
                            new_edge_dict['source_name'].append(st1)
                            new_edge_dict['target_name'].append(st2)
                            new_edge_dict['source_ID'].append(st_id1)
                            new_edge_dict['target_ID'].append(st_id2)
                            new_edge_dict['distance'].append(merged_distance)
                            new_edge_dict['geometry'].append(merged_geom)

                # create new gdf            
                new_edge_gdf = gpd.GeoDataFrame(new_edge_dict)

                # merge with original edge list (if original edge list is empty, replace with new edge list)
                if edge_list.empty:
                    edge_list = new_edge_gdf
                else:
                    edge_list = pd.concat([edge_list, new_edge_gdf], ignore_index = True)

            # set the characteristics
            edge_list.insert(0, 'N05_003', op)
            edge_list.insert(0, 'N05_002', line)

            # add to the list of edge lists
            edge_dfs.append(edge_list)

        # finished company
        print(f'Finished {op}')

    # combine all edges
    edge_list = gpd.GeoDataFrame(pd.concat(edge_dfs, ignore_index = True))
    
    # return edge list
    return edge_list

# create transfer edge list
def transfer_edge_list(stations, distance = 200, minutes = 10):
    '''
    Creates transfer edge links between stations that are within a distance and not operated by the same line.

    Parameters
    ----------
    stations : geopandas.GeoDataFrame
        GeoDataFrame of stations that transfer links.
    distance : int
        The maximum distance (in meters) that two stations will be considered transferrable.
    minutes : int
        The time for which it takes for the transfer.

    Returns
    -------
    transfers : geopandas.GeoDataFrame
        GeoDataFrame of edge list of the transfers.
    '''
    # get only the required columns
    stations_ext = stations[['N05_002', 'N05_003', 'N05_006', 'N05_011', 'geometry']].copy()

    # get buffer
    stations_buffer = stations_ext.copy()
    stations_buffer['geometry'] = stations_ext.buffer(distance)
    
    # get points within buffer and not identical
    transfers = stations_buffer.sjoin(stations_ext, predicate = 'contains')
    transfers = transfers[(transfers['N05_003_left'] != transfers['N05_003_right']) | (transfers['N05_002_left'] != transfers['N05_002_right'])]

    # get only the required columns
    transfers = transfers[['N05_006_left', 'N05_006_right']].copy().rename(columns = {'N05_006_left': 'source_id', 'N05_006_right': 'target_id'})

    # add time,
    transfers['time'] = minutes
    # add status
    transfers['type'] = 'rail_transfer'

    return transfers

# get edges between transport and street
def transit_links(transit_network, street_network, minutes = (10, 0.1)):
    '''
    Creates edges between the transit layer and the street layer.

    Parameters
    ----------
    transit_network : networkx.Graph
        The graph that has the network for the transit nodes
    steet_network : networkx.Graph
        The street network generated by osmnx
    minutes : (float, float)
        The time it takes to ride and get off the network (in minutes)
    
    Returns
    -------
    link_network : networkx.graph
        A graph containing transit links between the two layers.
    '''
    
    # create link network
    link_network = nx.MultiDiGraph()

    # copy the networks
    G1 = transit_network.copy()
    G2 = street_network.copy()

    # add name to network
    link_network.graph.update({'name': f'link_{G1.name}_{G2.name}'})

    # relabel nodes to include name of graph
    nx.relabel_nodes(G1, {node_id: f'{G1.name}_{node_id}' for node_id in list(G1.nodes)}, copy=False)
    nx.relabel_nodes(G2, {node_id: f'{G2.name}_{node_id}' for node_id in list(G2.nodes)}, copy=False)

    for stop in G1.nodes(data = True):
        # get name and geometry of stop
        stop_name = stop[0]
        stop_geometry = stop[1]['geometry']
        x = stop_geometry.x
        y = stop_geometry.y

        # get nearest node and distance for 
        nearest_node, node_distance = ox.distance.nearest_nodes(G2, x, y, return_dist = True)

        # add transfer link to the transfer network
        link_network.add_edge(stop_name, nearest_node, time = minutes[0], distance = node_distance, layer = 'transfer')
        link_network.add_edge(nearest_node, stop_name, time = minutes[1], distance = node_distance, layer = 'transfer')

    return link_network

# define function that creates the multi-layer network
def create_multilayer_network(layers, transfers):
    '''
    Creates the multi-layer network from networks of layers and transfers.

    Parameters
    ----------
    layers : list of networkx.Graph
        list of graphs that constitute each layer in the multilayer network.
    transfers : list of networkx.Graph
        List of graphs that constitute transfer layers 

    Returns
    -------
    G_multilayer : networkx.Graph
        The multilayer graph    
    '''

    # create copy of the networks
    G_layers = [G.copy() for G in layers]
    G_transfers = [G.copy() for G in transfers]

    for G in G_layers:
        # rename the node names so that it includes the layers
        nx.relabel_nodes(G, {node_id: f'{G.name}_{node_id}' for node_id in list(G.nodes)}, copy=False)
        # add attribute of layer to each node and edge
        nx.set_node_attributes(G, G.name, 'layer')
        nx.set_edge_attributes(G, G.name, 'layer')
 
    # combine the layers together
    G_multilayer = nx.union_all(G_layers)

    # add the edges of the transfer to this edge
    for G in G_transfers:
        G_multilayer.add_edges_from(G.edges(data = True))

    return G_multilayer   

def find_nearest_node(G_nodes, point):

    return node, distance
