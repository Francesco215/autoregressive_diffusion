function createCustomLink(anchorText, anchorLink) {
    var aTag = document.createElement("a");
    aTag.href = anchorLink;
    aTag.innerHTML = anchorText;

    return aTag;
}

function createCustomText(anchorText, type = "h2") {
    var Class;
    if (type == "h2") Class = "section";
    if (type == "h3") Class = "subsection";

    var tag = document.createElement(type);
    tag.innerHTML = anchorText;
    tag.setAttribute("class", Class);

    return tag;
}

//this code defines and manages the sections
class Sec extends HTMLElement {
    connectedCallback() {
        var text = this.textContent;
        this.textContent = "";
        this.appendChild(createCustomText(text, "h2"));

    };
}
customElements.define('d-section', Sec);


//this code defines and manages the sub-sections
class SubSec extends HTMLElement {
    connectedCallback() {
        var text = this.textContent;
        this.textContent = "";
        this.appendChild(createCustomText(text, "h3"));

    };
}
customElements.define('d-subsection', SubSec);





// this is to assign the numbers to the sections
// var sections = document.querySelectorAll('.section,.subsection');
// var n_section = 0;
// var n_subsection = 0;
// for (var i = 0; i < sections.length; i++) {
//     var parent = sections[i].parentElement;
//     if (sections[i].className == "section") {
//         n_section++;
//         n_subsection = 0;
//         parent.setAttribute("number", n_section);
//         sections[i].innerHTML = n_section + " " + sections[i].innerHTML;
//     }
//     if (sections[i].className == "subsection") {
//         n_subsection++;
//         parent.setAttribute("number", n_section + "." + n_subsection);
//         sections[i].innerHTML = n_section + "." + n_subsection + " " + sections[i].innerHTML;
//     }
// }

class Equation extends HTMLElement {
    connectedCallback() {
        this.setAttribute("class", "equation");
    };
}
customElements.define('d-equation', Equation);


//this code manages the references
class Ref extends HTMLElement {
    connectedCallback() {
        var key = this.getAttribute("key");
        var element = document.getElementById(key)
        var number = element.getAttribute('number');

        if (element.getAttribute('class') == "equation") { //this part is only valid if it is an equation
            var equation = element;
            for (i = 0; i < 10; i++) { //find the element with katex inside
                if (equation.firstElementChild.getAttribute('class') == "katex") { break; }
                equation = equation.firstElementChild;
            }

            var hover_box = document.createElement("d-hover-box"); //create the hover box
            hover_box.innerHTML = equation.innerHTML; //put the equation inside
            hover_box.firstElementChild.style.padding = "10px"; //and add some padding

            this.after(hover_box);
            this.onmouseover = function () { hover_box.style.display = "block"; }
            this.onmouseout = function () { hover_box.style.display = "none"; }
        }
        this.appendChild(createCustomLink(number, "#" + key));

    };
}


//this code manages the equations,
document.addEventListener("DOMContentLoaded", function () {
    var equations = document.querySelectorAll('.equation');
    for (var i = 0; i < equations.length; i++)
        equations[i].setAttribute("number", i + 1);

    customElements.define('d-reference', Ref); //this line calls the reference manager

});