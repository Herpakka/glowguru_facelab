class Score:
    # มีจุดบนหน้าผาก sc_fore
    # ใช้posteriseแล้วก็ยังมีจุด sc_foreP
    # มีจุดบนแก้ม sc_cheek
    # จุดเกาะกลุ่มบนหน้าผาก sc_foreD
    # จุดกระจายบนหน้าผาก sc_foreS
    # ใช้posteriseแล้วจุดลด sc_decreaseP
    # ใช้posteriseแล้วจุดเกาะกลุ่ม sc_denseP
    
    # 1:0 2:1.5 3:3 4:6
    
    # 19.5  มัน
    # 10.5  ปกติ
    # 18    แห้ง
    def __init__ (self):
        sc1 = 0
        sc2 = 0
        sc3 = 0
        sc4 = False
        sc5 = False
        sc6 = False
        sc7 = False
    def sc_fore(self, oil_all):
        return 1 if oil_all > 10 else 2 if oil_all > 50 else 0

    def sc_foreP(self, oil_allP):
        return 1 if oil_allP > 10 else 2 if oil_allP > 50 else 0
    
    def sc_cheek(self, oil_all):
        return 1 if oil_all > 10 else 2 if oil_all > 50 else 0

    def sc_foreD(self, fore_dense):
        return fore_dense[0] > 0

    def sc_foreS(self, fore_dense):
        return fore_dense[1] > 0
  
    def sc_decreaseP(self, count_before_poster, count_after_poster):
        return count_after_poster > count_before_poster
        
    def sc_denseP(self,fore_dense,PZfore_dense):
        return fore_dense[0] > PZfore_dense[0]
        