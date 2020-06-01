# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:05 2020

@author: paulg
"""

import xlsxwriter

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('Expenses01.xlsx')
worksheet = workbook.add_worksheet('top')
worksheet2 = workbook.add_worksheet('side')


# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

title = ['Name', 'Area','Perimeter','length','height']

# create the name of the file 
for t in (title):
    worksheet.write(row, col,t)

    col += 1

# Write a total using a formula.


workbook.close()

title = ['Name', 'Area','Perimeter','length','height']
