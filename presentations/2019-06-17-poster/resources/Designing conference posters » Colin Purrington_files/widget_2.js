//KO-Fi.com
//13/02/2016
//version 2 - Removed need to user [] because of issues in Wordpress
var kofiwidget2 = kofiwidget2 ||
(function() {
    var style = "";
    var html = "";
    var color = "";
    var text = "";
    var id = "";

    return {
        init: function(pText, pColor, pId) {
            //init
            color = pColor;
            text = pText;
            id = pId;
            style =
                "img.kofiimg{display: initial!important;vertical-align:middle;height:13px!important;width:20px!important;padding-top:0;padding-bottom:0;border:none;margin-top:0;margin-right:5px;margin-left:0;margin-bottom:4px;content:url('https://ko-fi.com/img/cuplogo.svg')}.kofiimg:after{vertical-align:middle;height:25px;padding-top:0;padding-bottom:0;border:none;margin-top:0;margin-right:6px;margin-left:0;margin-bottom:4px;content:url('https://ko-fi.com/img/whitelogo.svg')}.btn-container{display:inline-block!important;white-space:nowrap;min-width:160px}span.kofitext{color:#fff !important;letter-spacing: -0.15px!important;text-wrap:none;vertical-align:middle;line-height:33px !important;padding:0;text-align:center;text-decoration:none!important; text-shadow: 0 1px 1px rgba(34, 34, 34, 0.3);}.kofitext a{color:#fff !important;text-decoration:none:important;}.kofitext a:hover{color:#fff !important;text-decoration:none}a.kofi-button{box-shadow: 1px 2px 1px rgba(0, 0, 0, 0.2);line-height:32px!important;min-width:150px;display:inline-block!important;background-color:#46b798;padding:2px 11px !important;text-align:center !important;border-radius:3px;color:#fff;cursor:pointer;overflow-wrap:break-word;vertical-align:middle;border:0 none #fff !important;font-family:'Quicksand',Helvetica,Century Gothic,sans-serif !important;text-decoration:none;text-shadow:none;font-weight:700!important;font-size:14px !important}a.kofi-button:visited{color:#fff !important;text-decoration:none !important}a.kofi-button:hover{opacity:.85;color:#f5f5f5 !important;text-decoration:none !important}a.kofi-button:active{color:#f5f5f5 !important;text-decoration:none !important}.kofitext img.kofiimg {height:13px!important;width:20px!important;display: initial;-webkit-filter: drop-shadow(0 1px 1px rgba(34, 34, 34, 0.3));}";
            style = "<style>" + style + "</style>";
            html =
                "<link href='https://fonts.googleapis.com/css?family=Quicksand:400,700' rel='stylesheet' type='text/css'>";
            html +=
                '<div class=btn-container><a title="Support me on ko-fi.com" class="kofi-button" style="background-color:[color];" href="http://ko-fi.com/[id]" target="_blank"> <span class="kofitext"><img src="https://ko-fi.com/img/cuplogo.svg" class="kofiimg"/>[text]</span></a></div>';
        },
        getHTML: function() {
            var rtn = style + html.replace("[color]", color).replace("[text]", text).replace("[id]", id);
            return rtn;
        },
        draw: function() {
            document.writeln(style + html.replace("[color]", color).replace("[text]", text).replace("[id]", id));
        }
    };
}());