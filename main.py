#from func import *
import time
import sys
import os
import numpy as np
import requests
from BTrees.OOBTree import OOBTree
from collections import defaultdict
import re



inFile = sys.argv[1]

Var_dict = {} # a dictionary containing all tables

def main(op_file):
		'''
    This function reads in an txt file containing operations, parses the file and carries out required commands in order.
    Input:
        op_file: the name of txt file containing operations
    Output:
        None
    Influence on global environment:
        This function add entries (new tables) to Var_dict in global environment.
    '''

		if op_file[-4:] != '.txt':
				op_file += '.txt'

		funcs = {'inputfromfile': inputfromfile, 'select': select, 'project': project, 'sort': sort, 'concat': concat, 'join': join, 'sumgroup': sumgroup, 'avggroup': avggroup, 'countgroup': countgroup, 'sum': sum_, 'avg': avg, 'count': count, 'Btree': Btree, 'Hash': Hash, 'movsum': movsum, 'movavg': movavg, 'outputtofile': outputtofile}

		comment = re.compile(r'^[ ]+//')
		assign = re.compile(r':=')

		with open(op_file) as f:
				d = f.readlines()
				d = [i for i in d if not comment.search(i)]
				d = [i.split('//')[0].strip() for i in d]
    
    
		filename = 'AllOperations.txt' # save the outputs

		for op in d:
				if assign.search(op):
						t_name, t_command = (i.strip() for i in op.split(':='))
				else:
						t_name, t_command = None, op
				a = t_command.split(',')
				command = []
				for i in range(len(a)):
						if i == 0:
								command.extend(a[i].split('('))
								if re.search(r'\)$', command[1]):
										command[1] = command[1][:-1]
						elif i == len(a)-1:
								command.append(a[i][:-1].strip())
						else:
								command.append(a[i].strip())

				start = time.process_time()
				if t_name:
						Var_dict[t_name] = funcs[command[0]](command[1:], Var_dict)
				else:
						funcs[command[0]](command[1:], Var_dict)
				print(time.process_time() - start)

				if t_name:
						if os.path.exists(filename):
								append_write = 'a' # append if already exists
						else:
								append_write = 'w' # make a new file if not

						alloperations = open(filename, append_write)
						alloperations.write(t_name + '\n')
						alloperations.write(t_command + '\n')

						table, header_dict = Var_dict[t_name]
						#print(header_dict)
						#print(table)
						header = list(header_dict.keys())
						for i in range(len(header)):
								if i < len(header)-1:
										alloperations.write(str(header[i]) + '|')
								elif i == len(header)-1:
										alloperations.write(str(header[i]) + '\n')
						for row in table:
								for i in range(len(row)):
										if i < len(row)-1:
												alloperations.write(str(row[i]) + '|')
										elif i == len(row)-1:
												alloperations.write(str(row[i]) + '\n')
						alloperations.close()

def inputfromfile(args, Var_dict):
		'''
		This function imports and cleans txt files containing tables.
		Input:
				args: a list containing name of txt files.
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				[table, header_index]: a list containing content and header of imported table.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''
		file_name = args[0]
		if file_name[-4:] != '.txt':
				file_name += '.txt'


		with open(file_name) as f:
				d = f.readlines()
				d = [i.strip().split('|') for i in d]
				headers = d[0]
				d = d[1:]

		table = []  
		for r in d:
				ri = []
				for i in r:
						try:
								ri.append(int(i))
						except:
								ri.append(i)

				table.append(ri)
    
		header_index = {}
		for i in range(len(headers)):
				header_index[headers[i]] = i
        
		return [table, header_index]


def join(args, Var_dict):
		'''
		This function joins two tables based on condition. We firstly parse the condition by the function called 'parseCondition', then extracted 
		rows that satisfy the condition by calling the function 'checkCond'.
		'''
		T1 = Var_dict[ args[0] ][0]
		T1_header_index = Var_dict[ args[0] ][1]

		T2 = Var_dict[ args[1] ][0]
		T2_header_index = Var_dict[ args[1] ][1]

		conditions = parseCondition(args[2])

		Table = []

		for rowA in T1:
				for rowB in T2:
						flg = True
						for c in conditions:
								if not checkCond(rowA, rowB, c, T1_header_index, T2_header_index):
										flg = False
										break
						if flg:
								Table.append(rowA + rowB)

		T1_name = args[0]
		T2_name = args[1]
    
		T1_new_headername = [T1_name+'_'+i for i in T1_header_index]
		T2_new_headername = [T2_name+'_'+i for i in T2_header_index]
		new_header = T1_new_headername+T2_new_headername
    
		header_index = {}
		for i in range(len(new_header)):
				header_index[new_header[i]] = i
    
		return [Table,header_index]

    

def sumgroup(args, Var_dict):
		'''
		This function sums the column by group. We treat the columns for grouping as composite key. Then extracted 
		rows according to the key and take sum.
		'''
		T, T_header_index = Var_dict[args[0]]
		index_sum = T_header_index[args[1]]
		index_group=[T_header_index[i] for i in args[2:]]
    
		grouper = defaultdict(float) 

		for row in T:
				groupVal = [row[i] for i in  index_group]
				groupVal= tuple(groupVal)

				grouper[groupVal] += row[index_sum]

		resultTable= []

		for k in grouper.keys():
				newRow = [ grouper[k] ] + list(k)
				resultTable.append(newRow)
    
		new_header=['sum'+'('+args[1]+')']+args[2:]
		header_index = {}
		for i in range(len(new_header)):
				header_index[new_header[i]] = i
    
		return [resultTable, header_index]



def avggroup(args, Var_dict):
		'''
		This function average the column by group. We treat the columns for grouping as composite key. Then extracted 
		rows according to the key and take average.
		'''
		T, T_header_index = Var_dict[args[0]]
		index_avg = T_header_index[args[1]]
		index_group=[T_header_index[i] for i in args[2:]]
    
		grouper = defaultdict(float) 
		counter=defaultdict(int) 

		for row in T:
				groupVal = [row[i] for i in  index_group]
				groupVal= tuple(groupVal)

				grouper[groupVal] += row[index_avg]
				counter[groupVal] += 1
    
		resultTable= []

		for k in grouper.keys():
				newRow = [ grouper[k]/counter[k] ] + list(k)
				resultTable.append(newRow)

		new_header=['avg'+'('+args[1]+')']+args[2:]
		header_index = {}
		for i in range(len(new_header)):
				header_index[new_header[i]] = i
		return [resultTable, header_index]

def countgroup(args, Var_dict):
		'''
		This function groups instances and count the number of instances in each group.
		Input:
				args: a list containing name of table, name of column we want to group and count and name of column(s) according to which we group our table
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				a list containing content and header of table after countgroup.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''

		T, T_header_index = Var_dict[args[0]]
		index_count = T_header_index[args[1]]
		index_group=[T_header_index[i] for i in args[2:]]
     
		counter=defaultdict(int) 

		for row in T:
				groupVal = [row[i] for i in  index_group]
				groupVal= tuple(groupVal)

				counter[groupVal] += 1
    
		resultTable= []

		for k in grouper.keys():
				newRow = [ counter[k] ] + list(k)
				resultTable.append(newRow)
        
		new_header=['count'+'('+args[1]+')']+args[2:]
		header_index = {}
		for i in range(len(new_header)):
				header_index[new_header[i]] = i
        
		return [resultTable, header_index]



def sum_(args, Var_dict): 
		'''
		This function sums the column. 
		'''
		T, T_header_index = Var_dict[ args[0]]
		index = T_header_index[args[1]]
    
		L=[]
		for row in T:
				L.append(row[index])
    
		new_header='sum'+'('+args[1]+')'
		header_index = {}
		header_index[new_header] = 0
		return [[[sum(L)]],header_index]



def avg(args, Var_dict):
		'''
		This function average the column. 
		'''
		T, T_header_index = Var_dict[ args[0]]
		index = T_header_index[args[1]]
    
		L = []
		for row in T:
				L.append(row[index])
    
		new_header='avg'+'('+args[1]+')'
		header_index = {}
		header_index[new_header] = 0
		return [[[sum(L)/len(L)]],header_index]

def count(args, Var_dict):
		'''
		This function counts the number of instances in a table.
		Input:
				args: a list containing name of table, name of column we want to count the number of instances in.
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				a list containing content and header of table after count.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''

		T, T_header_index = Var_dict[ args[0]]
		#index = T_header_index[args[1]]
    
		#L = []
		#for row in T:
				#L.append(row[index])
        
		#new_header='count'+'('+args[1]+')'
		new_header='count'+'('+args[0]+')'
		header_index = {}
		header_index[new_header] = 0
		#return [len(L), header_index]
		return [[[len(T)]], header_index]

    

def parseCondition(arg):
		'''
		This function is used to parse the condition. It is called by the function 'join'.
		'''
		argList = arg.split('and')
		argList = [i.strip() for i in argList]
		argList = [i.strip('(') for i in argList]
		argList = [i.strip(')') for i in argList]
    

		arithops = {'=': np.equal,'!=': np.not_equal, '<': np.less, '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal}
		cond = []
		for ar in argList:
				for c in arithops.keys():
						if c in ar:
								twoTb = ar.split(c)

								twoTb[0] = twoTb[0].strip()
								first = twoTb[0].split('.')

								twoTb[1] = twoTb[1].strip()
								second = twoTb[1].split('.')
								operands = first + second
								operands.append(arithops[c])
								cond.append(operands)
								break

		return cond


def checkCond(row1,row2,condi, T1_header_index ,T2_header_index):
		'''
		This function is used to check the condition. It is called by the function 'join'.
		'''
		indx1 = T1_header_index[condi[1]]
		indx2 = T2_header_index[condi[3]]
		return condi[4] (row1[indx1], row2[indx2] )


def Btree(args, Var_dict):
		'''
		This function initialize the Btree index.
		'''
		T1 = Var_dict[ args[0] ][0]
		T1_header_index = Var_dict[ args[0] ][1]

		colNum = T1_header_index[args[1]]
		if len(Var_dict[ args[0] ]) <= 2:
				Var_dict[ args[0] ].append( { args[1] : OOBTree() } )
		else:
				Var_dict[ args[0] ][2][ args[1] ] = OOBTree()

		T1_fast_stuct = Var_dict[ args[0] ][2]

		for i in range(len(T1)):
				row = T1[i]
				if row[colNum] in T1_fast_stuct[ args[1] ]:
						T1_fast_stuct[ args[1] ][ row[colNum] ].add(i)
				else:
						T1_fast_stuct[ args[1] ][ row[colNum] ] = set([i])


def Hash(args, Var_dict):
		'''
		This function initialize the Hash index.
		'''
		T1 = Var_dict[ args[0] ][0]
		T1_header_index = Var_dict[ args[0] ][1]

		colNum = T1_header_index[args[1]]
		if len(Var_dict[ args[0] ]) <= 2:
				Var_dict[ args[0] ].append( { args[1] : {} } )
		else:
				Var_dict[ args[0] ][2][ args[1] ] = {}

		T1_fast_stuct = Var_dict[ args[0] ][2]

		for i in range(len(T1)):
				row = T1[i]
				if row[colNum] in T1_fast_stuct[ args[1] ]:
						T1_fast_stuct[ args[1] ][ row[colNum] ].add(i)
				else:
						T1_fast_stuct[ args[1] ][ row[colNum] ] = set([i])



def select(args, Var_dict):
		'''
		This function selects rows based on condition. We firstly parse the condition by the function called 'Condi_operations', then extracted 
		rows that satisfy the condition by calling the function 'get_valid_rows'. If the Btree or Hash index is added before calling the 'select' function, the speed is much quicker than before.
		'''
		T1 = Var_dict[ args[0] ][0]
		T1_header_index = Var_dict[ args[0] ][1]
		fast_struct = {}

		if len(Var_dict[ args[0] ])>2:
				fast_struct = Var_dict[ args[0] ][2]

		conditions = space_filter(args[1])
		orCondi = conditions.split('or')
		orCondi = [i.strip('(') for i in orCondi]
		orCondi = [i.strip(')') for i in orCondi]

		resRow = set()

		for orc in orCondi:
				andCondi = orc.split('and')
				andCondi = [i.strip('(') for i in andCondi]
				andCondi = [i.strip(')') for i in andCondi]
				andCondi = Condi_operations(andCondi)

				goodRow = get_valid_rows(T1, T1_header_index, fast_struct, andCondi )

				resRow = resRow.union(goodRow)


		finalArrayTb =[T1[j] for j in resRow]
    
		return [finalArrayTb,T1_header_index]



def Condi_operations(condi):
		'''
		This function is used to parse the condition for selection. It is called by select function.
		'''
		XX = ['!=','<=','>=','=','<','>']

		res = []

		for c in condi:
				for ops in XX:
						if ops in c:
								col, val = c.split(ops)
								if val.isnumeric():
										val = int(val)

								res.append([col, ops, val])

								break
		return res


def get_valid_rows(tb, tbCol, structs, condi):
		'''
		This function is used to extracted rows that satisfy the condition for selection. IIt is called by select function.
		'''

		arithops = {'=': np.equal,'!=': np.not_equal, '<': np.less, '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal}
		candidate = range(len(tb))

		res = []

		for c in condi:
				if c[0] in structs and c[1] == '=':
						#print(structs[c[0]])
						candidate = structs[c[0]][c[2]]

		for i in candidate:
				flg = True
				for C in condi:
						check = arithops[C[1]]( tb[i][ tbCol[C[0]] ], C[2] )
						if not check:
								flg = False
								break
				if flg:
						res.append(i)

		return set(res)



def space_filter(S):
		'''
		This function is used to remove the whitespace in a command. It is called by select function.
		'''
		rt = ''
		for c in S:
				if c != ' ':
						rt += c
		return rt


def project(args, Var_dict):
		'''
		This function extracts columns we are interested in.
		Input:
				args: a list containing name of table, name of column we want to extract
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				a list containing content and header of table after project.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''

		table = Var_dict[ args[0] ][0]
		header_index = Var_dict[ args[0] ][1]
    
    # consider project all
		if len(args) == 1:
				return [table, header_index]
    
		headers_new =[]
		table_new = []
		for col in args[1:]:
				index = header_index[col]
				headers_new.append(col)
				table_new.append([row[index] for row in table])
    
		header_index = {}
		for i in range(len(headers_new)):
				header_index[headers_new[i]] = i
        
		return [table_new, header_index]

def sort(args, Var_dict):
		'''
		This function sorts table based on one or more columns.
		Input:
				args: a list containing name of table, name(s) of column(s) we sort the table baed on.
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				a list containing content and header of table after sort.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''

		table = Var_dict[ args[0] ][0]
		header_index = Var_dict[ args[0] ][1]
    
		cols_sort_by = args[1:]
		sort_by_index = np.lexsort([[int(row[header_index[col]]) for row in table] for col in cols_sort_by])
		table = np.array(table)
		table = table[sort_by_index].tolist()
    
		return [table, header_index]

def concat(args, Var_dict):
		'''
		This function concatenates tables with same header.
		Input:
				args: a list containing names of tables we want to concatenate
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				a list containing content and header of table after concat.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''

		table_1 = Var_dict[ args[0] ][0]
		table_2 = Var_dict[ args[1] ][0]
		header_index = Var_dict[ args[0] ][1]
        
		return [table_1 + table_2, header_index]

def movsum(args, Var_dict):
		'''
		This function calculates moving sum of a column of a table.
		Input:
				args: a list containing name of table, name(s) of column(s) we want to apply moving sum and the number n which implies n-way moving sum
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				a list containing content and header of table after movsum.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''
    
		table = Var_dict[ args[0] ][0]
		table_header_index = Var_dict[ args[0] ][1]
		cols, num = args[1:-1], int(args[-1])

		def movsum_helper(col):
				col_movsum = []
				while len(col_movsum) < num:
						col_movsum.append(sum(col[:len(col_movsum)+1]))
				while len(col_movsum) < len(col):
						col_movsum.append(sum(col[len(col_movsum)+1-num:len(col_movsum)+1]))
				return col_movsum

		col_movsum = []
		header_index = {}
		i = 0
		for col in cols:
				index = table_header_index[col]
				column = [int(row[index]) for row in table]
				col_movsum.append(movsum_helper(column))
				header_index[col] = i
				i += 1
        
		return [col_movsum, header_index]

def movavg(args, Var_dict):
		'''
		This function calculates moving average of a column of a table.
		Input:
				args: a list containing name of table, name(s) of column(s) we want to apply moving average and the number n which implies n-way moving average
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				a list containing content and header of table after movavg.
		Influence on global environment:
				This function adds an entry (new table) to Var_dict in global environment.
		'''
    
		table = Var_dict[ args[0] ][0]
		table_header_index = Var_dict[ args[0] ][1]
		cols, num = args[1:-1], int(args[-1])

		def movavg_helper(col):
				col_movavg = []
				while len(col_movavg) < num:
						col_movavg.append(sum(col[:(len(col_movavg)+1)]) / len(col[:(len(col_movavg)+1)]))
				while len(col_movavg) < len(col):
						col_movavg.append(sum(col[len(col_movavg)+1-num:len(col_movavg)+1]) / len(col[len(col_movavg)+1-num:len(col_movavg)+1]))
				return col_movavg
    
		col_movavg = []
		header_index = {}
		i = 0
		for col in cols:
				index = table_header_index[col]
				column = [int(row[index]) for row in table]
				col_movavg.append(movavg_helper(column))
				header_index[col] = i
				i += 1
        
		return [col_movavg, header_index]



def outputtofile(args, Var_dict):
		'''
		This function exports table to txt file.
		Input:
				args: a list containing name of table to export and name of txt file
				Var_dict: a dictionary containing existed tables. (key: table name; value: table content and header)
		Output:
				None
		Influence on global environment:
				None
		'''

		table_name, output_name = args[0], args[1]
		table, header_dict = Var_dict[table_name]
		header = list(header_dict.keys())
    
		with open('ym1970_jz3741_' + output_name + '.txt', 'w') as f:
				for i in range(len(header)):
						if i < len(header)-1:
								f.write(str(header[i]) + '|')
						elif i == len(header)-1:
								f.write(str(header[i]) + '\n')
				for row in table:
						for i in range(len(row)):
								if i < len(row)-1:
										f.write(str(row[i]) + '|')
								elif i == len(row)-1:
										f.write(str(row[i]) + '\n')


main(inFile)