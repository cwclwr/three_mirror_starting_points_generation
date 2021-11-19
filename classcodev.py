import numpy as np
import win32com
from win32com.client import Dispatch


class codev:

    cvserver=win32com.client.Dispatch('CODEV.Command.114')

    def cvon(self):

        print('CODE V is starting...')
        self.cvserver.SetStartingDirectory('C:\CVUSER')
        self.cvserver.StartCodeV()
        print('CODE V version', self.cvserver.GetCodeVVersion(), 'is now running')

        return

    def cvoff(self):
        self.cvserver.StopCodeV()
        print('CODE V server has been closed')

        return

    def cvopen(self,filename):
        self.cvserver.command('res '+filename)

        return

    def cvsave(self,filename):
        self.cvserver.command('sav '+filename)

        return

    def cvcmd(self,s):
        result = self.cvserver.command(s)

        return

    def cvbuf(self, bufnum, startrow, endrow, startcol, endcol):
        temp = np.zeros((endrow - startrow + 1, endcol - startcol + 1))
        # temp = tuple(temp)
        # bufout = self.cvserver.BufferToArray(startrow, endrow, startcol, endcol, bufnum, temp)
        bufout = self.cvserver.BUFFER_TO_ARRAY(bufnum, temp, startrow, endrow, startcol, endcol, 0)
        if bufout[0] != 0:
            print('Data transfer failed!')

        buf_listtype = list(bufout[1])
        for c in bufout[1]:
            buf_listtype[bufout[1].index(c)] = list(c)

        return buf_listtype


    def cvmacro(self, s, args=None):

        if args is None:
            args = []
        s_='in "'+s+'"'
        for a in args:
            s_ += ' '+str(a)
        self.cvserver.command(s_)
        return
