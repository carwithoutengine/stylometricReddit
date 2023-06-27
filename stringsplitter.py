#0 replace commas, hyphens etc. with whitespace
#1 split sentence and store in list
#2 make all lowercase
#3 copy list contents into a set to eliminate duplicates
#4 reconvert this back into a list
#5 sort alphabetically
#6 use this new list to count frequency in original list and store stuff in dictionary

       
src="""Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
       Nullam placerat mi vel nisi ullamcorper, eget efficitur 
       risus semper. Morbi ultrices nibh non felis vestibulum, 
       non pellentesque leo posuere. Quisque at dui pretium, 
       vestibulum sapien ac, eleifend tellus. Sed tristique, 
       velit in imperdiet gravida, nunc turpis vulputate velit, 
       nec hendrerit leo nibh eget justo. Nulla facilisi. Sed id 
       lectus ut quam tempor consectetur. Suspendisse malesuada, 
       mi vel efficitur semper, ante lorem rhoncus nisi, a lacinia 
       tortor lacus quis eros. Sed et libero ipsum. Nulla facilisi. 
       Donec sit amet metus non eros tristique sodales. Vivamus 
       dictum arcu eget blandit lobortis. Sed efficitur bibendum 
       nulla eu finibus. Sed sed ex risus."""

print(src)

clean=src
clean=clean.replace(', ',' ')
clean=clean.replace('. ',' ')
clean=clean.replace('.',' ')
clean=clean.replace('-',' ')
clean=clean.replace('! ',' ')
clean=clean.lower()

print(clean)

raw=clean.split()
#print(raw)
unique=raw
#print(unique)
unique=set(unique)
#print(unique)
unique=list(unique)
#print(unique)
unique.sort()
print(unique)


final=dict()

for x in unique:
    final[x]=raw.count(x)

print(final)
