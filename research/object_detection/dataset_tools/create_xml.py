width  = 2448
height = 2048

i = 0
for image in images:

    xml_file = open(str(i)+".xml","w+")
    xml_file.write("<annotation><filename>"+str(i)+".png</filename>\
                   <size><width>"+str(width)+"</width><height>"+str(height)+"</height>\
                   <depth>3</depth></size>\
                   <object><pose>Frontal</pose>\
                   <truncated>0</truncated>\
                   <difficult>0</difficult></object></annotation>")
    xml_file.close()
