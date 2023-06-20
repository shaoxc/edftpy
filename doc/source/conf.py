import sys
import sphinx_rtd_theme
project = 'eDFTpy'
copyright = '2021, Pavanello Research Group'
author = 'Pavanello Research Group'

release = '0.0.1'

source_suffix = '.rst'
master_doc = 'index'

nbsphinx_execute = 'never'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.graphviz',
              'nbsphinx',
              'sphinx_panels']

templates_path = ['templates']
exclude_patterns = ['build']

html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_bootstrap_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_favicon = 'static/edftpy.ico'
# html_logo = 'static/edftpy.png'
html_logo = 'static/edftpy.svg'
html_style = 'custom.css'
html_static_path = ['static']
html_last_updated_fmt = '%A, %d %b %Y %H:%M:%S'

html_theme_options = {
    'logo_only': True,
    'prev_next_buttons_location': 'both',
    # 'style_nav_header_background' : '#E67E22'
    # 'style_nav_header_background' : '#27AE60'
    'style_nav_header_background' : '#bdc3c7'
}

# latex_show_urls = 'inline'
latex_show_pagerefs = True
latex_documents = [('index', not True)]

graphviz_output_format = 'svg'


#Add external links to source code
def linkcode_resolve(domain, info):
    print('info module', info)
    if domain != 'py' or not info['module']:
        return None

    filename = info['module'].replace('.', '/')+'.py'
    return "https ://gitlab.com/pavanello-research-group/edftpy/tree/master/%s" % filename
