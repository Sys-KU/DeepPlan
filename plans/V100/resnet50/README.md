|Layer                    |Initial approach         |DeepPlan (DHA)
|-------------------------|-------------------------|-------------------------
|0-Conv2d             (0.036 MB) |O                          |O
|1-BatchNorm2d        (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|2-ReLU               (0.000 MB) |X                          |X
|3-MaxPool2d          (0.000 MB) |X                          |X
|4-Conv2d             (0.016 MB) |O                          |O
|5-BatchNorm2d        (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|6-ReLU               (0.000 MB) |X                          |X
|7-Conv2d             (0.141 MB) |X (direct-host-access)     |X (direct-host-access)
|8-BatchNorm2d        (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|9-ReLU               (0.000 MB) |X                          |X
|10-Conv2d            (0.062 MB) |O                          |O
|11-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|12-Conv2d            (0.062 MB) |O                          |O
|13-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|14-ReLU              (0.000 MB) |X                          |X
|15-Conv2d            (0.062 MB) |O                          |O
|16-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|17-ReLU              (0.000 MB) |X                          |X
|18-Conv2d            (0.141 MB) |X (direct-host-access)     |X (direct-host-access)
|19-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|20-ReLU              (0.000 MB) |X                          |X
|21-Conv2d            (0.062 MB) |O                          |O
|22-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|23-ReLU              (0.000 MB) |X                          |X
|24-Conv2d            (0.062 MB) |O                          |O
|25-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|26-ReLU              (0.000 MB) |X                          |X
|27-Conv2d            (0.141 MB) |X (direct-host-access)     |X (direct-host-access)
|28-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |X (direct-host-access)
|29-ReLU              (0.000 MB) |X                          |X
|30-Conv2d            (0.062 MB) |O                          |O
|31-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|32-ReLU              (0.000 MB) |X                          |X
|33-Conv2d            (0.125 MB) |O                          |O
|34-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|35-ReLU              (0.000 MB) |X                          |X
|36-Conv2d            (0.562 MB) |X (direct-host-access)     |X (direct-host-access)
|37-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|38-ReLU              (0.000 MB) |X                          |X
|39-Conv2d            (0.250 MB) |O                          |O
|40-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|41-Conv2d            (0.500 MB) |O                          |O
|42-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|43-ReLU              (0.000 MB) |X                          |X
|44-Conv2d            (0.250 MB) |X (direct-host-access)     |O
|45-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|46-ReLU              (0.000 MB) |X                          |X
|47-Conv2d            (0.562 MB) |X (direct-host-access)     |X (direct-host-access)
|48-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|49-ReLU              (0.000 MB) |X                          |X
|50-Conv2d            (0.250 MB) |O                          |O
|51-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|52-ReLU              (0.000 MB) |X                          |X
|53-Conv2d            (0.250 MB) |X (direct-host-access)     |O
|54-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|55-ReLU              (0.000 MB) |X                          |X
|56-Conv2d            (0.562 MB) |X (direct-host-access)     |X (direct-host-access)
|57-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|58-ReLU              (0.000 MB) |X                          |X
|59-Conv2d            (0.250 MB) |O                          |O
|60-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|61-ReLU              (0.000 MB) |X                          |X
|62-Conv2d            (0.250 MB) |X (direct-host-access)     |O
|63-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|64-ReLU              (0.000 MB) |X                          |X
|65-Conv2d            (0.562 MB) |X (direct-host-access)     |X (direct-host-access)
|66-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|67-ReLU              (0.000 MB) |X                          |X
|68-Conv2d            (0.250 MB) |O                          |O
|69-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|70-ReLU              (0.000 MB) |X                          |X
|71-Conv2d            (0.500 MB) |O                          |O
|72-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|73-ReLU              (0.000 MB) |X                          |X
|74-Conv2d            (2.250 MB) |O                          |O
|75-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|76-ReLU              (0.000 MB) |X                          |X
|77-Conv2d            (1.000 MB) |X (direct-host-access)     |O
|78-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |O
|79-Conv2d            (2.000 MB) |O                          |O
|80-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |O
|81-ReLU              (0.000 MB) |X                          |X
|82-Conv2d            (1.000 MB) |X (direct-host-access)     |O
|83-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|84-ReLU              (0.000 MB) |X                          |X
|85-Conv2d            (2.250 MB) |O                          |O
|86-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|87-ReLU              (0.000 MB) |X                          |X
|88-Conv2d            (1.000 MB) |X (direct-host-access)     |O
|89-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |X (direct-host-access)
|90-ReLU              (0.000 MB) |X                          |X
|91-Conv2d            (1.000 MB) |X (direct-host-access)     |O
|92-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|93-ReLU              (0.000 MB) |X                          |X
|94-Conv2d            (2.250 MB) |O                          |O
|95-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|96-ReLU              (0.000 MB) |X                          |X
|97-Conv2d            (1.000 MB) |X (direct-host-access)     |O
|98-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |O
|99-ReLU              (0.000 MB) |X                          |X
|100-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|101-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|102-ReLU             (0.000 MB) |X                          |X
|103-Conv2d           (2.250 MB) |O                          |O
|104-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|105-ReLU             (0.000 MB) |X                          |X
|106-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|107-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|108-ReLU             (0.000 MB) |X                          |X
|109-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|110-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|111-ReLU             (0.000 MB) |X                          |X
|112-Conv2d           (2.250 MB) |O                          |O
|113-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|114-ReLU             (0.000 MB) |X                          |X
|115-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|116-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|117-ReLU             (0.000 MB) |X                          |X
|118-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|119-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|120-ReLU             (0.000 MB) |X                          |X
|121-Conv2d           (2.250 MB) |O                          |O
|122-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|123-ReLU             (0.000 MB) |X                          |X
|124-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|125-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|126-ReLU             (0.000 MB) |X                          |X
|127-Conv2d           (2.000 MB) |X (direct-host-access)     |O
|128-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|129-ReLU             (0.000 MB) |X                          |X
|130-Conv2d           (9.000 MB) |O                          |O
|131-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|132-ReLU             (0.000 MB) |X                          |X
|133-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|134-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|135-Conv2d           (8.000 MB) |X (direct-host-access)     |O
|136-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|137-ReLU             (0.000 MB) |X                          |X
|138-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|139-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|140-ReLU             (0.000 MB) |X                          |X
|141-Conv2d           (9.000 MB) |O                          |O
|142-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|143-ReLU             (0.000 MB) |X                          |X
|144-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|145-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|146-ReLU             (0.000 MB) |X                          |X
|147-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|148-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|149-ReLU             (0.000 MB) |X                          |X
|150-Conv2d           (9.000 MB) |O                          |O
|151-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|152-ReLU             (0.000 MB) |X                          |X
|153-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|154-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|155-ReLU             (0.000 MB) |X                          |X
|156-AdaptiveAvgPool2d (0.000 MB) |X                          |X
|157-Linear           (7.816 MB) |X (direct-host-access)     |O
