import xlwings as xw
import numpy as np

class Excel(object):
    def __init__(self, file_path = None):
        # Initial the parameter of excel
        print("Initial the parameter of excel.")
        self.app = xw.App(visible=True, add_book=False)
        if file_path == None:
            self.wb = self.app.books.add()
        else:
            self.wb = self.app.books.open(file_path)
        self.sheet = self.wb.sheets[0]

    def new_sheet(self, name):
        self.wb.sheets.add(name= name)

    def read_excel(self, start):
        return self.sheet.range(start).value

    def write_excel(self, start, content, vertical=False):
        print('write the content.\n')
        self.sheet.range(start).options(transpose=vertical).value = content

    def save_excel(self, file_name):
        print('save the excel file.')
        self.wb.save(file_name)

    def close_excel(self):
        self.wb.close()
        self.app.quit()

    def clear_excel(self):
        self.sheet.clear()

    def write_loss_and_iou(self, record, loss, val_loss, iou, avr_iou):
        self.write_excel('a1', record)
        self.write_excel('a2', 'loss')
        self.write_excel('b2', 'val_loss')
        self.write_excel('a3', loss, vertical=True)
        self.write_excel('b3', val_loss, vertical=True)
        self.write_excel('c2', 'iou')
        self.write_excel('d2', 'avr_iou')
        self.write_excel('c3', iou, vertical=True)
        self.write_excel('d3', avr_iou, vertical=True)

if __name__ == "__main__":
    loss_record_file = ".\\result\\data\\V3.1&3.2_test\\model_CE2.xlsx"
    iou_v1 = ".\\result\\data\\V3.1&3.2_test\\V3.1"
    iou_v2 = ".\\result\\data\\V3.1&3.2_test\\V3.2"
    iou_file_name = "20200218_256(50%)_14191_V3.1&3.2_UNet(2Dense4)_bin_iou.xlsx"
    order1 = "q"
    order2 = "r"

    record = Excel(file_path= loss_record_file)
    iou1 = Excel(file_path= iou_v1 + "\\" + iou_file_name)
    iou2 = Excel(file_path=iou_v2 + "\\" + iou_file_name)

    iou1_value = iou1.read_excel("c3:c1767")
    iou1.close_excel()
    iou2_value = iou2.read_excel("c3:c1785")
    iou2.close_excel()

    record.write_excel(start= order1 + "1", content="UNet(2Dense4)")
    avr_iou = (np.sum(iou1_value) + np.sum(iou2_value)) / (len(iou1_value) + len(iou2_value))
    record.write_excel(start= order1 + "2", content="avr_iou")
    record.write_excel(start= order1 + "3", content="V3.1_iou")
    record.write_excel(start= order1 + "4", content="V3.2_iou")
    record.write_excel(start= order1 + "5", content="V3.1")
    record.write_excel(start= order1 + "6", content=iou1_value, vertical=True)
    record.write_excel(start= order2 + "2", content=avr_iou)
    record.write_excel(start= order2 + "3", content=(np.sum(iou1_value) / len(iou1_value)))
    record.write_excel(start= order2 + "4", content=(np.sum(iou2_value) / len(iou2_value)))
    record.write_excel(start= order2 + "5", content="V3.2")
    record.write_excel(start= order2 + "6", content=iou2_value, vertical=True)

    record.save_excel(file_name = None)
    record.close_excel()


