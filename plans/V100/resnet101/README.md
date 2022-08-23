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
|22-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |O
|23-ReLU              (0.000 MB) |X                          |X
|24-Conv2d            (0.062 MB) |O                          |O
|25-BatchNorm2d       (0.001 MB) |X (direct-host-access)     |O
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
|40-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |O
|41-Conv2d            (0.500 MB) |O                          |O
|42-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|43-ReLU              (0.000 MB) |X                          |X
|44-Conv2d            (0.250 MB) |X (direct-host-access)     |O
|45-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|46-ReLU              (0.000 MB) |X                          |X
|47-Conv2d            (0.562 MB) |X (direct-host-access)     |O
|48-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|49-ReLU              (0.000 MB) |X                          |X
|50-Conv2d            (0.250 MB) |O                          |O
|51-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |O
|52-ReLU              (0.000 MB) |X                          |X
|53-Conv2d            (0.250 MB) |X (direct-host-access)     |O
|54-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|55-ReLU              (0.000 MB) |X                          |X
|56-Conv2d            (0.562 MB) |X (direct-host-access)     |O
|57-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|58-ReLU              (0.000 MB) |X                          |X
|59-Conv2d            (0.250 MB) |O                          |O
|60-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|61-ReLU              (0.000 MB) |X                          |X
|62-Conv2d            (0.250 MB) |X (direct-host-access)     |O
|63-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|64-ReLU              (0.000 MB) |X                          |X
|65-Conv2d            (0.562 MB) |X (direct-host-access)     |O
|66-BatchNorm2d       (0.002 MB) |X (direct-host-access)     |X (direct-host-access)
|67-ReLU              (0.000 MB) |X                          |X
|68-Conv2d            (0.250 MB) |O                          |O
|69-BatchNorm2d       (0.010 MB) |X (direct-host-access)     |X (direct-host-access)
|70-ReLU              (0.000 MB) |X                          |X
|71-Conv2d            (0.500 MB) |O                          |O
|72-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|73-ReLU              (0.000 MB) |X                          |X
|74-Conv2d            (2.250 MB) |X (direct-host-access)     |O
|75-BatchNorm2d       (0.005 MB) |X (direct-host-access)     |O
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
|89-BatchNorm2d       (0.020 MB) |X (direct-host-access)     |O
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
|119-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |O
|120-ReLU             (0.000 MB) |X                          |X
|121-Conv2d           (2.250 MB) |O                          |O
|122-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|123-ReLU             (0.000 MB) |X                          |X
|124-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|125-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|126-ReLU             (0.000 MB) |X                          |X
|127-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|128-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|129-ReLU             (0.000 MB) |X                          |X
|130-Conv2d           (2.250 MB) |O                          |O
|131-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|132-ReLU             (0.000 MB) |X                          |X
|133-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|134-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|135-ReLU             (0.000 MB) |X                          |X
|136-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|137-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|138-ReLU             (0.000 MB) |X                          |X
|139-Conv2d           (2.250 MB) |O                          |O
|140-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|141-ReLU             (0.000 MB) |X                          |X
|142-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|143-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|144-ReLU             (0.000 MB) |X                          |X
|145-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|146-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|147-ReLU             (0.000 MB) |X                          |X
|148-Conv2d           (2.250 MB) |O                          |O
|149-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|150-ReLU             (0.000 MB) |X                          |X
|151-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|152-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|153-ReLU             (0.000 MB) |X                          |X
|154-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|155-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|156-ReLU             (0.000 MB) |X                          |X
|157-Conv2d           (2.250 MB) |O                          |O
|158-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|159-ReLU             (0.000 MB) |X                          |X
|160-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|161-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|162-ReLU             (0.000 MB) |X                          |X
|163-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|164-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|165-ReLU             (0.000 MB) |X                          |X
|166-Conv2d           (2.250 MB) |O                          |O
|167-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|168-ReLU             (0.000 MB) |X                          |X
|169-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|170-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|171-ReLU             (0.000 MB) |X                          |X
|172-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|173-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|174-ReLU             (0.000 MB) |X                          |X
|175-Conv2d           (2.250 MB) |O                          |O
|176-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|177-ReLU             (0.000 MB) |X                          |X
|178-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|179-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|180-ReLU             (0.000 MB) |X                          |X
|181-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|182-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|183-ReLU             (0.000 MB) |X                          |X
|184-Conv2d           (2.250 MB) |O                          |O
|185-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|186-ReLU             (0.000 MB) |X                          |X
|187-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|188-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|189-ReLU             (0.000 MB) |X                          |X
|190-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|191-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |O
|192-ReLU             (0.000 MB) |X                          |X
|193-Conv2d           (2.250 MB) |O                          |O
|194-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|195-ReLU             (0.000 MB) |X                          |X
|196-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|197-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|198-ReLU             (0.000 MB) |X                          |X
|199-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|200-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|201-ReLU             (0.000 MB) |X                          |X
|202-Conv2d           (2.250 MB) |O                          |O
|203-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|204-ReLU             (0.000 MB) |X                          |X
|205-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|206-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|207-ReLU             (0.000 MB) |X                          |X
|208-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|209-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|210-ReLU             (0.000 MB) |X                          |X
|211-Conv2d           (2.250 MB) |O                          |O
|212-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|213-ReLU             (0.000 MB) |X                          |X
|214-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|215-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|216-ReLU             (0.000 MB) |X                          |X
|217-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|218-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|219-ReLU             (0.000 MB) |X                          |X
|220-Conv2d           (2.250 MB) |O                          |O
|221-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|222-ReLU             (0.000 MB) |X                          |X
|223-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|224-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|225-ReLU             (0.000 MB) |X                          |X
|226-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|227-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|228-ReLU             (0.000 MB) |X                          |X
|229-Conv2d           (2.250 MB) |O                          |O
|230-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|231-ReLU             (0.000 MB) |X                          |X
|232-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|233-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|234-ReLU             (0.000 MB) |X                          |X
|235-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|236-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|237-ReLU             (0.000 MB) |X                          |X
|238-Conv2d           (2.250 MB) |O                          |O
|239-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|240-ReLU             (0.000 MB) |X                          |X
|241-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|242-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|243-ReLU             (0.000 MB) |X                          |X
|244-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|245-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|246-ReLU             (0.000 MB) |X                          |X
|247-Conv2d           (2.250 MB) |O                          |O
|248-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|249-ReLU             (0.000 MB) |X                          |X
|250-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|251-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|252-ReLU             (0.000 MB) |X                          |X
|253-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|254-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|255-ReLU             (0.000 MB) |X                          |X
|256-Conv2d           (2.250 MB) |O                          |O
|257-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|258-ReLU             (0.000 MB) |X                          |X
|259-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|260-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|261-ReLU             (0.000 MB) |X                          |X
|262-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|263-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|264-ReLU             (0.000 MB) |X                          |X
|265-Conv2d           (2.250 MB) |O                          |O
|266-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|267-ReLU             (0.000 MB) |X                          |X
|268-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|269-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|270-ReLU             (0.000 MB) |X                          |X
|271-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|272-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|273-ReLU             (0.000 MB) |X                          |X
|274-Conv2d           (2.250 MB) |O                          |O
|275-BatchNorm2d      (0.005 MB) |X (direct-host-access)     |X (direct-host-access)
|276-ReLU             (0.000 MB) |X                          |X
|277-Conv2d           (1.000 MB) |X (direct-host-access)     |O
|278-BatchNorm2d      (0.020 MB) |X (direct-host-access)     |O
|279-ReLU             (0.000 MB) |X                          |X
|280-Conv2d           (2.000 MB) |X (direct-host-access)     |O
|281-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|282-ReLU             (0.000 MB) |X                          |X
|283-Conv2d           (9.000 MB) |O                          |O
|284-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|285-ReLU             (0.000 MB) |X                          |X
|286-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|287-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|288-Conv2d           (8.000 MB) |X (direct-host-access)     |O
|289-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|290-ReLU             (0.000 MB) |X                          |X
|291-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|292-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|293-ReLU             (0.000 MB) |X                          |X
|294-Conv2d           (9.000 MB) |O                          |O
|295-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|296-ReLU             (0.000 MB) |X                          |X
|297-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|298-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|299-ReLU             (0.000 MB) |X                          |X
|300-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|301-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|302-ReLU             (0.000 MB) |X                          |X
|303-Conv2d           (9.000 MB) |O                          |O
|304-BatchNorm2d      (0.010 MB) |X (direct-host-access)     |O
|305-ReLU             (0.000 MB) |X                          |X
|306-Conv2d           (4.000 MB) |X (direct-host-access)     |O
|307-BatchNorm2d      (0.039 MB) |X (direct-host-access)     |O
|308-ReLU             (0.000 MB) |X                          |X
|309-AdaptiveAvgPool2d (0.000 MB) |X                          |X
|310-Linear           (7.816 MB) |X (direct-host-access)     |O
