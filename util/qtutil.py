#-*- coding:utf-8 -*-

import io
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage

def copy_fig_to_clipboard(fig, **args):
    buf = io.BytesIO()
    fig.savefig(buf, **args)
    QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
    buf.close()
